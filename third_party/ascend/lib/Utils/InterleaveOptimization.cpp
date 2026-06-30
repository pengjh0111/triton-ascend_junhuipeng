/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/Utils/InterleaveOptimization.h"
#include "ascend/include/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <utility>

namespace mlir {
namespace triton {
// For origin MemRefType of ReinterpretCastOp under interleave state, here wanna
// adjust its shape info by expanding last dimension double.
MemRefType expandInterleaveMemRefType(MemRefType originType) {
  // Double the last dimension shape
  SmallVector<int64_t> shape(originType.getShape());
  shape.back() = shape.back() * 2;

  // Adjuest layout attribute
  StridedLayoutAttr originLayout =
      llvm::dyn_cast<StridedLayoutAttr>(originType.getLayout());
  // If offset is static, just reset it to 0
  auto offset = originLayout.getOffset() == ShapedType::kDynamic
                    ? originLayout.getOffset()
                    : 0;
  // Set last dimension stride to 1
  SmallVector<int64_t> stride(originLayout.getStrides());
  stride.back() = 1;

  return MemRefType::get(
      shape, originType.getElementType(),
      StridedLayoutAttr::get(originType.getContext(), offset, stride));
}

// *********************
// **      NOTE       **
// *********************
// How to determine new offset is a little tricky and specific
// Here just consider this state in triton language:
//
// dim_range = tl.arange(0, BLOCK // 2)
// last_dim_even_range = dim_range * 2
// last_dim_odd_range = dim_range * 2 + 1
//
// Here `multiply two` represents that last dimension stride is 2, and
// `add constant one` represents whether it's odd index part of
// deinterleave result.
//
// Therefore, how to distinguish interleave/deinterleave on even index or odd
// index is whether last dimension range explicitly `add constant one` without
// any other operation. In IR it's shown that whether defining op of
// `castOffset` is an arith::addOp, as this arith::addOp would contain above
// `add constant one` opeartion after LegacyAddPtrConverter.
//
// Well, index mode should be passed to interleave/deinterleave, in other words,
// `add constant one` should work on offset of next insert_slice/extract_slic.
// The new reinterpretcast just wanna describe whole tensor, so new castOffset
// is just from non-last diemsnion accumulation and remove `add constant one`
bool checkIsCaseOffsetValid(OpFoldResult originOffset) {
  // If offset is constant int(IndexAttr), the int value could only be 0 or 1
  // if offset is a value from add constant operation and not from `add constant
  // one` operation, it's invalid.
  if (llvm::isa<Attribute>(originOffset)) {
    int64_t intOffset = getConstantIntValue(originOffset).value();
    return intOffset == 0 || intOffset == 1;
  } else if (llvm::isa<Value>(originOffset)) {
    auto op = cast<Value>(originOffset).getDefiningOp();
    if (op && llvm::isa<arith::AddIOp>(op)) {
      if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
        if (auto constLHS = addOp.getLhs().getDefiningOp<arith::ConstantOp>()) {
          return dyn_cast<IntegerAttr>(constLHS.getValueAttr()).getInt() == 1;
        }
        if (auto constRHS = addOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
          return dyn_cast<IntegerAttr>(constRHS.getValueAttr()).getInt() == 1;
        }
      }
    }
  }
  return true;
}

std::pair<OpFoldResult, IndexMode>
recountReinterpretCastOffset(OpFoldResult originOffset, Builder &builder) {
  // To trace value type offset
  std::function<bool(Operation *)> traceOffset = [&](Operation *op) -> bool {
    // Consider constant one in `add constant one` operation
    if (llvm::isa<arith::ConstantOp>(op))
      return false;

    if (llvm::isa<arith::AddIOp>(op)) {
      auto addOp = llvm::cast<arith::AddIOp>(op);
      if (auto constLHS = addOp.getLhs().getDefiningOp<arith::ConstantOp>()) {
        assert(dyn_cast<IntegerAttr>(constLHS.getValueAttr()).getInt() == 1 &&
               "Arith::constant value of addi's operand must be 1 when "
               "calculate deinterleave offset");
        return false;
      }
      if (auto constRHS = addOp.getRhs().getDefiningOp<arith::ConstantOp>()) {
        assert(dyn_cast<IntegerAttr>(constRHS.getValueAttr()).getInt() == 1 &&
               "Arith::constant value of addi's operand must be 1 when "
               "calculate deinterleave offset");
        return false;
      }
    }
    return true;
  };

  IndexMode evenOrOdd = IndexMode::EVEN_MODE;
  // Reuse origin offset if there's no 'add constant one'
  OpFoldResult newOffset = originOffset;
  if (llvm::isa<Attribute>(originOffset)) {
    // If offset is constant int(IndexAttr),
    // the int value could only be 0 or 1
    int64_t intOffset = getConstantIntValue(originOffset).value();
    assert((intOffset == 0 || intOffset == 1));
    if (intOffset == 1) {
      evenOrOdd = IndexMode::ODD_MODE;
      newOffset = builder.getIndexAttr(0);
    }
  } else if (llvm::isa<Value>(originOffset)) {
    if (!traceOffset(cast<Value>(originOffset).getDefiningOp())) {
      evenOrOdd = IndexMode::ODD_MODE;
      Operation *traceResult = findFirstMatchingOperandDef(
          cast<Value>(originOffset).getDefiningOp(), traceOffset);
      assert(traceResult->getNumResults() == 1 &&
             "Offset defining operation must have one result");
      newOffset = traceResult->getResult(0);
    }
  }

  return {newOffset, evenOrOdd};
}

LogicalResult
DeinterleaveStatusOptimization(triton::LoadOp op,
                               triton::LoadOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter) {
  auto ptr = adaptor.getPtr();
  if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto loc = op.getLoc();

    // 1. Get new source memref type
    auto srcType = expandInterleaveMemRefType(reinterpretCast.getType());

    // 2. Create new ReinterpretCastOp
    auto originCastOffset = reinterpretCast.getConstifiedMixedOffset();
    auto castSize = reinterpretCast.getConstifiedMixedSizes();
    auto castStride = reinterpretCast.getConstifiedMixedStrides();
    // Actually, `castSize` is always constant value as `MemRefType` result
    if (auto lastDimSize = getConstantIntValue(castSize.back())) {
      castSize.back() = rewriter.getIndexAttr(lastDimSize.value() * 2);
    } else {
      return failure();
    }
    // Last element of castStride is also constant value as prerequisite
    // is that last dimension stride of casted memref type is always 2.
    castStride.back() = rewriter.getIndexAttr(1);
    if (!checkIsCaseOffsetValid(originCastOffset)) {
      return failure();
    }
    auto [castOffset, indexMode] =
        recountReinterpretCastOffset(originCastOffset, rewriter);
    auto newCastOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, srcType, reinterpretCast.getViewSource(), castOffset, castSize,
        castStride);

    // 3. Load via hfusion.load (src is newCastOp viewed as tensor, dst is empty)
    Value srcT = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(srcType.getShape(), srcType.getElementType()),
        newCastOp.getResult(), /*restrict=*/true, /*writable=*/false);
    Value dstEmpty = rewriter.create<tensor::EmptyOp>(
        loc, srcType.getShape(), srcType.getElementType());
    Value newTensor =
        rewriter
            .create<hfusion::LoadOp>(
                loc,
                TypeRange{RankedTensorType::get(srcType.getShape(),
                                                srcType.getElementType())},
                ValueRange{srcT}, ValueRange{dstEmpty})
            ->getResult(0);

    // 5. Implement tensor extract_slice to represent deinterleave
    // Here use `castOffset` to determine whether even index deinterleave or
    // odd index.
    SmallVector<OpFoldResult> extractOffsets(srcType.getRank(),
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> extractSizes = llvm::to_vector(
        llvm::map_range(srcType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    // Adjust extract_slice shape
    switch (indexMode) {
    case IndexMode::EVEN_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(0);
      break;
    case IndexMode::ODD_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(1);
      break;
    }
    extractStrides.back() = rewriter.getIndexAttr(2);
    extractSizes.back() = rewriter.getIndexAttr(srcType.getShape().back() / 2);

    Value deinterleaveSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newTensor, extractOffsets, extractSizes, extractStrides);

    rewriter.replaceOp(op, deinterleaveSlice);
    return success();
  }

  return failure();
}

LogicalResult DeinterleaveStatusWithMaskOptimization(
    triton::LoadOp op, triton::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, MaskState &mstate) {
  auto ptr = adaptor.getPtr();
  if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto loc = op.getLoc();

    // 1. Get new source memref type
    auto srcType = expandInterleaveMemRefType(reinterpretCast.getType());

    // 2. Create new ReinterpretCastOp
    auto originCastOffset = reinterpretCast.getConstifiedMixedOffset();
    auto castSize = reinterpretCast.getConstifiedMixedSizes();
    auto castStride = reinterpretCast.getConstifiedMixedStrides();

    if (auto lastDimSize = getConstantIntValue(castSize.back())) {
      castSize.back() = rewriter.getIndexAttr(lastDimSize.value() * 2);
    } else {
      return failure();
    }
    castStride.back() = rewriter.getIndexAttr(1);
    if (!checkIsCaseOffsetValid(originCastOffset)) {
      return failure();
    }
    auto [castOffset, indexMode] =
        recountReinterpretCastOffset(originCastOffset, rewriter);

    auto newCastOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, srcType, reinterpretCast.getViewSource(), castOffset, castSize,
        castStride);

    // 3. Compute subview sizes (last dim doubled for expanded interleave type)
    SmallVector<OpFoldResult> subviewStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> subviewOffsets = mstate.offsets;
    SmallVector<OpFoldResult> subviewSizes = mstate.dims;
    std::optional<int64_t> originSubviewLastDim =
        getConstantIntValue(subviewSizes.back());
    assert(originSubviewLastDim.has_value());
    subviewSizes.back() =
        rewriter.getIndexAttr(originSubviewLastDim.value() * 2);

    // 4. Build full dst tensor; conditionally fill with other if present.
    // pad_load is NOT used here: the deinterleave mask guarantees the last dim
    // is always full (entry guard), so left/right padding would be 0. Any row
    // direction clipping (non-last dims) must be handled by fill+insert_slice.
    Value fullDst = rewriter.create<tensor::EmptyOp>(
        loc, srcType.getShape(), srcType.getElementType());
    auto other = op.getOther();
    if (other) {
      Value otherScalar =
          mlir::ConverterUtils::getScalarValue(other, loc, rewriter);
      assert(otherScalar && "other value used in masked load produced by "
                            "unsupported instruction!");
      fullDst = mlir::ConverterUtils::buildConditionalFillTensor(
          rewriter, loc, otherScalar, fullDst, subviewSizes);
    }

    // 5. hfusion.load on subview of newCastOp, insert_slice into fullDst.
    auto argSubviewType = memref::SubViewOp::inferResultType(
        srcType, subviewOffsets, subviewSizes, subviewStrides);
    memref::SubViewOp srcSubview = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(argSubviewType), newCastOp, subviewOffsets,
        subviewSizes, subviewStrides);
    auto srcSubviewShape =
        llvm::cast<MemRefType>(srcSubview.getType()).getShape();
    Value srcSubT = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get(srcSubviewShape, srcType.getElementType()),
        srcSubview.getResult(), /*restrict=*/true, /*writable=*/false);
    Value loadDst = rewriter.create<tensor::EmptyOp>(
        loc, subviewSizes, srcType.getElementType());
    auto hfuseLoadTy = llvm::cast<RankedTensorType>(loadDst.getType());
    Value loadedSlice =
        rewriter
            .create<hfusion::LoadOp>(loc, TypeRange{hfuseLoadTy},
                                     ValueRange{srcSubT}, ValueRange{loadDst})
            ->getResult(0);
    SmallVector<OpFoldResult> unitStrides(srcType.getRank(),
                                          rewriter.getIndexAttr(1));
    Value newTensor = rewriter.create<tensor::InsertSliceOp>(
        loc, loadedSlice, fullDst, subviewOffsets, subviewSizes, unitStrides);

    // 6. Implement tensor extract_slice to represent deinterleave
    // Here use `castOffset` to determine whether even index deinterleave or
    // odd index.
    SmallVector<OpFoldResult> extractOffsets(srcType.getRank(),
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> extractSizes = llvm::to_vector(
        llvm::map_range(srcType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    switch (indexMode) {
    case IndexMode::EVEN_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(0);
      break;
    case IndexMode::ODD_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(1);
      break;
    }
    extractStrides.back() = rewriter.getIndexAttr(2);
    extractSizes.back() = rewriter.getIndexAttr(srcType.getShape().back() / 2);

    Value deinterleaveSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newTensor, extractOffsets, extractSizes, extractStrides);

    rewriter.replaceOp(op, deinterleaveSlice);
    return success();
  }
  return failure();
}

LogicalResult
InterleaveStatusOptimization(SmallVector<Operation *> materializeVec) {
  OpBuilder builder(materializeVec[1]);
  auto loc = materializeVec[1]->getLoc();

  // With memref-level store, outs is the destination memref directly.
  auto getStoreDestMemref = [](hfusion::StoreOp storeOp) -> Value {
    return storeOp.getOutputs().front();
  };

  // With memref-level store, ins is ToBufferOp(srcTensor); unwrap to get the
  // original tensor for use in tensor::InsertSliceOp.
  auto getStoreSrcTensor = [](hfusion::StoreOp storeOp) -> Value {
    Value src = storeOp.getInputs().front();
    if (auto toBufferOp = src.getDefiningOp<bufferization::ToBufferOp>())
      src = toBufferOp.getOperand();
    return src;
  };

  auto firstReinterpretCastOp =
      getStoreDestMemref(llvm::cast<hfusion::StoreOp>(materializeVec[0]))
          .getDefiningOp<memref::ReinterpretCastOp>();
  auto secondReinterpretCastOp =
      getStoreDestMemref(llvm::cast<hfusion::StoreOp>(materializeVec[1]))
          .getDefiningOp<memref::ReinterpretCastOp>();

  assert(firstReinterpretCastOp && secondReinterpretCastOp);

  // Judge whether two `ReinterpretCastOp` shape satisfy interleave state
  // a. both size are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedSizes(),
          secondReinterpretCastOp.getConstifiedMixedSizes())) {
    return failure();
  }
  // b. both strides are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedStrides(),
          secondReinterpretCastOp.getConstifiedMixedStrides())) {
    return failure();
  }
  // c. both offsets should satisfy tricky rule
  auto firstOriginCastOffset =
      firstReinterpretCastOp.getConstifiedMixedOffset();
  auto secondOriginCastOffset =
      secondReinterpretCastOp.getConstifiedMixedOffset();
  if (!checkIsCaseOffsetValid(firstOriginCastOffset) ||
      !checkIsCaseOffsetValid(secondOriginCastOffset)) {
    return failure();
  }

  std::pair<IndexMode, IndexMode> indexModeRecord;
  OpFoldResult newCastOffset;
  if (llvm::isa<Attribute>(firstOriginCastOffset) &&
      llvm::isa<Attribute>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^ static_cast<int>(secondIndexMode)))
      return failure();
    newCastOffset = builder.getIndexAttr(0);
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else if (llvm::isa<Value>(firstOriginCastOffset) &&
             llvm::isa<Value>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^
          static_cast<int>(secondIndexMode)) ||
        (llvm::dyn_cast<Value>(firstCastOffset) !=
         llvm::dyn_cast<Value>(secondCastOffset)))
      return failure();

    if (firstIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(firstCastOffset);
    }
    if (secondIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(secondCastOffset);
    }
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else {
    return failure();
  }

  // Create new op
  // 1. Get new destination memref type
  auto dstType = expandInterleaveMemRefType(firstReinterpretCastOp.getType());

  // 2. New tensor::EmptyOp
  auto emptyTensor = builder.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

  // 3. New insert_slice from materialization source into new empty tensor
  SmallVector<OpFoldResult> insertOffsets(dstType.getRank(),
                                          builder.getIndexAttr(0));
  SmallVector<OpFoldResult> insertStrides(dstType.getRank(),
                                          builder.getIndexAttr(1));
  SmallVector<OpFoldResult> insertSizes = llvm::to_vector(
      llvm::map_range(dstType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return builder.getIndexAttr(dim);
      }));
  insertStrides.back() = builder.getIndexAttr(2);
  insertSizes.back() = builder.getIndexAttr(dstType.getShape().back() / 2);
  if (indexModeRecord.first == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertFirst = builder.create<tensor::InsertSliceOp>(
      loc,
      getStoreSrcTensor(llvm::cast<hfusion::StoreOp>(materializeVec[0])),
      emptyTensor.getResult(), insertOffsets, insertSizes, insertStrides);

  if (indexModeRecord.second == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertSecond = builder.create<tensor::InsertSliceOp>(
      loc,
      getStoreSrcTensor(llvm::cast<hfusion::StoreOp>(materializeVec[1])),
      insertFirst.getResult(), insertOffsets, insertSizes, insertStrides);

  // 4. Reinterpret_cast block arg
  auto newCastSize = firstReinterpretCastOp.getConstifiedMixedSizes();
  auto newCastStride = firstReinterpretCastOp.getConstifiedMixedStrides();
  newCastSize.back() = builder.getIndexAttr(dstType.getShape().back());
  newCastStride.back() = builder.getIndexAttr(1);
  auto newCastOp = builder.create<memref::ReinterpretCastOp>(
      loc, dstType, firstReinterpretCastOp.getViewSource(), newCastOffset,
      newCastSize, newCastStride);

  // 5. Create new single memref-level hfusion::StoreOp
  {
    auto srcTensor = insertSecond.getResult();
    auto dstMemref = newCastOp.getResult();
    auto srcTy = cast<RankedTensorType>(srcTensor.getType());
    auto srcMemrefTy =
        MemRefType::get(srcTy.getShape(), srcTy.getElementType());
    Value srcMemref =
        builder.create<bufferization::ToBufferOp>(loc, srcMemrefTy, srcTensor);
    builder.create<hfusion::StoreOp>(
        loc, TypeRange{}, ValueRange{srcMemref}, ValueRange{dstMemref});
  }

  // 6. Erase origin stores. Each origin op in materializeVec is a memref-level
  // hfusion.store with NO result and no users, so it can be erased directly.
  materializeVec[0]->erase();
  materializeVec[1]->erase();

  return success();
}

LogicalResult
InterleaveStatusWithMaskOptimization(SmallVector<Operation *> materializeVec) {
  OpBuilder builder(materializeVec[1]);

  // With memref-level store, outs is the destination memref directly.
  auto getStoreDestMemref = [](hfusion::StoreOp storeOp) -> Value {
    return storeOp.getOutputs().front();
  };

  // With memref-level store, ins is ToBufferOp(srcTensor); unwrap to get the
  // original tensor, then find the defining ExtractSliceOp.
  auto getStoreSrcExtractSlice =
      [](hfusion::StoreOp storeOp) -> tensor::ExtractSliceOp {
    Value src = storeOp.getInputs().front();
    if (auto toBufferOp = src.getDefiningOp<bufferization::ToBufferOp>())
      src = toBufferOp.getOperand();
    return src.getDefiningOp<tensor::ExtractSliceOp>();
  };

  auto firstSubviewOpOfReCast =
      getStoreDestMemref(llvm::cast<hfusion::StoreOp>(materializeVec[0]))
          .getDefiningOp<memref::SubViewOp>();
  auto firstSrcExtractSlice =
      getStoreSrcExtractSlice(llvm::cast<hfusion::StoreOp>(materializeVec[0]));
  auto firstReinterpretCastOp = firstSubviewOpOfReCast.getSource()
                                    .getDefiningOp<memref::ReinterpretCastOp>();

  auto secondSubviewOpOfReCast =
      getStoreDestMemref(llvm::cast<hfusion::StoreOp>(materializeVec[1]))
          .getDefiningOp<memref::SubViewOp>();
  auto secondSrcExtractSlice =
      getStoreSrcExtractSlice(llvm::cast<hfusion::StoreOp>(materializeVec[1]));
  auto secondReinterpretCastOp =
      secondSubviewOpOfReCast.getSource()
          .getDefiningOp<memref::ReinterpretCastOp>();

  // 1. Both source shapes of subview and extract_slice are equal
  if (firstSubviewOpOfReCast.getSourceType().getShape() !=
      firstSrcExtractSlice.getSourceType().getShape())
    return failure();
  if (secondSubviewOpOfReCast.getSourceType().getShape() !=
      secondSrcExtractSlice.getSourceType().getShape())
    return failure();
  if (firstSubviewOpOfReCast.getSourceType().getShape() !=
      secondSubviewOpOfReCast.getSourceType().getShape())
    return failure();

  // 2. both mask state are equal
  std::function<bool(OpFoldResult, OpFoldResult)> cmpFunc =
      mlir::isEqualConstantIntOrValue;
  if (!mlir::detail::sameOffsetsSizesAndStrides(firstSubviewOpOfReCast,
                                                firstSrcExtractSlice, cmpFunc))
    return failure();
  if (!mlir::detail::sameOffsetsSizesAndStrides(secondSubviewOpOfReCast,
                                                secondSrcExtractSlice, cmpFunc))
    return failure();
  if (!mlir::detail::sameOffsetsSizesAndStrides(
          firstSubviewOpOfReCast, secondSubviewOpOfReCast, cmpFunc))
    return failure();

  // 3. Still judge whether two `ReinterpretCastOp` shape satisfy request
  // a. both size are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedSizes(),
          secondReinterpretCastOp.getConstifiedMixedSizes()))
    return failure();
  // b. both strides are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedStrides(),
          secondReinterpretCastOp.getConstifiedMixedStrides()))
    return failure();
  // c. both offsets should satisfy tricky rule
  auto firstOriginCastOffset =
      firstReinterpretCastOp.getConstifiedMixedOffset();
  auto secondOriginCastOffset =
      secondReinterpretCastOp.getConstifiedMixedOffset();
  if (!checkIsCaseOffsetValid(firstOriginCastOffset) ||
      !checkIsCaseOffsetValid(secondOriginCastOffset)) {
    return failure();
  }

  std::pair<IndexMode, IndexMode> indexModeRecord;
  OpFoldResult newCastOffset;
  if (llvm::isa<Attribute>(firstOriginCastOffset) &&
      llvm::isa<Attribute>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^ static_cast<int>(secondIndexMode)))
      return failure();
    newCastOffset = builder.getIndexAttr(0);
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else if (llvm::isa<Value>(firstOriginCastOffset) &&
             llvm::isa<Value>(secondOriginCastOffset)) {
    auto [firstCastOffset, firstIndexMode] =
        recountReinterpretCastOffset(firstOriginCastOffset, builder);
    auto [secondCastOffset, secondIndexMode] =
        recountReinterpretCastOffset(secondOriginCastOffset, builder);

    if (!(static_cast<int>(firstIndexMode) ^
          static_cast<int>(secondIndexMode)) ||
        (llvm::dyn_cast<Value>(firstCastOffset) !=
         llvm::dyn_cast<Value>(secondCastOffset)))
      return failure();

    if (firstIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(firstCastOffset);
    }
    if (secondIndexMode == IndexMode::EVEN_MODE) {
      newCastOffset = llvm::dyn_cast<Value>(secondCastOffset);
    }
    indexModeRecord = {firstIndexMode, secondIndexMode};

  } else {
    return failure();
  }
  auto loc = materializeVec[1]->getLoc();

  // Create new op
  // 1. Get new destination memref type
  auto dstType = expandInterleaveMemRefType(firstReinterpretCastOp.getType());

  // 2. New tensor::EmptyOp
  auto emptyTensor = builder.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

  // 3. New insert_slice from extract_slice source into new empty tensor
  SmallVector<OpFoldResult> insertOffsets(dstType.getRank(),
                                          builder.getIndexAttr(0));
  SmallVector<OpFoldResult> insertStrides(dstType.getRank(),
                                          builder.getIndexAttr(1));
  SmallVector<OpFoldResult> insertSizes = llvm::to_vector(
      llvm::map_range(dstType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return builder.getIndexAttr(dim);
      }));
  insertStrides.back() = builder.getIndexAttr(2);
  insertSizes.back() = builder.getIndexAttr(dstType.getShape().back() / 2);
  if (indexModeRecord.first == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertFirst = builder.create<tensor::InsertSliceOp>(
      loc, firstSrcExtractSlice.getSource(), emptyTensor.getResult(),
      insertOffsets, insertSizes, insertStrides);

  if (indexModeRecord.second == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertSecond = builder.create<tensor::InsertSliceOp>(
      loc, secondSrcExtractSlice.getSource(), insertFirst.getResult(),
      insertOffsets, insertSizes, insertStrides);

  // 4. To enable store with mask, create new extract_slice
  SmallVector<OpFoldResult> extractOffsets =
      firstSrcExtractSlice.getMixedOffsets();
  SmallVector<OpFoldResult> extractStrides =
      firstSrcExtractSlice.getMixedStrides();
  SmallVector<OpFoldResult> extractSizes = firstSrcExtractSlice.getMixedSizes();
  if (!llvm::isa<Attribute>(extractSizes.back())) {
    return failure();
  }
  extractSizes.back() = builder.getIndexAttr(
      getConstantIntValue(extractSizes.back()).value() * 2);
  auto newSrcExtractSlice = builder.create<tensor::ExtractSliceOp>(
      loc, insertSecond.getResult(), extractOffsets, extractSizes,
      extractStrides);

  // 5. Reinterpret_cast block arg
  auto newCastSize = firstReinterpretCastOp.getConstifiedMixedSizes();
  auto newCastStride = firstReinterpretCastOp.getConstifiedMixedStrides();
  newCastSize.back() = builder.getIndexAttr(dstType.getShape().back());
  newCastStride.back() = builder.getIndexAttr(1);
  auto newCastOp = builder.create<memref::ReinterpretCastOp>(
      loc, dstType, firstReinterpretCastOp.getViewSource(), newCastOffset,
      newCastSize, newCastStride);

  // 6. Create new memref::SubViewOp of above new reinterpret_cast
  // Here could reuse shape info of new extract_slice
  auto dstSubviewType = memref::SubViewOp::inferResultType(
      dstType, extractOffsets, extractSizes, extractStrides);
  auto newSubviewOpOfReCast = builder.create<memref::SubViewOp>(
      loc, llvm::cast<MemRefType>(dstSubviewType), newCastOp, extractOffsets,
      extractSizes, extractStrides);

  // 7. Create new single memref-level hfusion::StoreOp
  {
    auto srcTensor = newSrcExtractSlice.getResult();
    auto dstMemref = newSubviewOpOfReCast.getResult();
    auto srcTy = cast<RankedTensorType>(srcTensor.getType());
    auto srcMemrefTy =
        MemRefType::get(srcTy.getShape(), srcTy.getElementType());
    Value srcMemref =
        builder.create<bufferization::ToBufferOp>(loc, srcMemrefTy, srcTensor);
    builder.create<hfusion::StoreOp>(
        loc, TypeRange{}, ValueRange{srcMemref}, ValueRange{dstMemref});
  }

  // 8. Erase origin operation. Each origin op is a memref-level hfusion.store
  // with NO result and no users, so it can be erased directly.
  materializeVec[0]->erase();
  materializeVec[1]->erase();
  if (firstSubviewOpOfReCast->use_empty()) {
    firstSubviewOpOfReCast->erase();
  }
  if (firstSrcExtractSlice->use_empty()) {
    firstSrcExtractSlice->erase();
  }
  if (secondSubviewOpOfReCast->use_empty()) {
    secondSubviewOpOfReCast->erase();
  }
  if (secondSrcExtractSlice->use_empty()) {
    secondSrcExtractSlice->erase();
  }

  return success();
}

} // namespace triton
} // namespace mlir
