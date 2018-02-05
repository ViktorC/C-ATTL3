/*
 * Dimensions.h
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef DIMENSIONS_H_
#define DIMENSIONS_H_

#include <cassert>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cattle {

/**
 * NOTE: Expression templates are absolutely unwarranted for such small objects (especially since they are very
 * unlikely to be used in any complex expressions). They have only been implemented for the sake of practical
 * learning.
 */

// Forward declarations.
template<typename IndexType, typename LhsExpr, typename OpType>
class UnaryDimExpression;
template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDimExpression;
template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDepthWiseDimExpression;

// Arithmetic operations.
template<typename Number> class SumOp {
public: inline static Number apply(Number lhs, Number rhs) { return lhs + rhs; };
};
template<typename Number> class SubOp {
public: inline static Number apply(Number lhs, Number rhs) { return lhs - rhs; };
};
template<typename Number> class MulOp {
public: inline static Number apply(Number lhs, Number rhs) { return lhs * rhs; };
};
template<typename Number> class DivOp {
public: inline static Number apply(Number lhs, Number rhs) { return lhs / rhs; };
};

template<typename Derived, typename IndexType>
class DimExpression {
	typedef DimExpression<Derived,IndexType> Self;
	template<typename OtherDerived> using Other = DimExpression<OtherDerived,IndexType>;
public:
	inline IndexType operator()(size_t i) const {
		return static_cast<const Derived&>(*this)(i);
	};
	inline IndexType get_height() const {
		return static_cast<const Derived&>(*this).get_height();
	};
	inline IndexType get_width() const {
		return static_cast<const Derived&>(*this).get_width();
	};
	inline IndexType get_depth() const {
		return static_cast<const Derived&>(*this).get_depth();
	};
	inline IndexType get_volume() const {
		return get_height() * get_width() * get_depth();
	};
	inline std::string to_string() const {
		return "[" + std::to_string(get_height()) + ", " + std::to_string(get_width()) + ", " +
				std::to_string(get_depth()) + "]";
	};
	template<typename OtherDerived>
	inline bool equals(const Other<OtherDerived>& dims) const {
		return get_height() == dims.get_height() && get_width() == dims.get_width() && get_depth() == dims.get_depth();
	};
	inline bool equals(IndexType height, IndexType width, IndexType depth) {
		return get_height() == height && get_width() == width && get_depth() == depth;
	};
	inline operator Derived&() {
		return static_cast<Derived&>(*this);
	};
	inline operator const Derived&() const {
		return static_cast<const Derived&>(*this);
	};
	template<typename OtherDerived>
	inline BinaryDepthWiseDimExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>
	add_along_depth(const Other<OtherDerived>& dims) {
		return BinaryDepthWiseDimExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDepthWiseDimExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>
	subtract_along_depth(const Other<OtherDerived>& dims) {
		return BinaryDepthWiseDimExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>
	operator+(const Other<OtherDerived>& dims) {
		return BinaryDimExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>
	operator-(const Other<OtherDerived>& dims) {
		return BinaryDimExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>(*this, dims);
	};
	inline UnaryDimExpression<IndexType,Self,SumOp<IndexType>>
	operator+(const IndexType& n) {
		return UnaryDimExpression<IndexType,Self,SumOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Self,SubOp<IndexType>>
	operator-(const IndexType& n) {
		return UnaryDimExpression<IndexType,Self,SubOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Self,DivOp<IndexType>>
	operator/(const IndexType& n) {
		return UnaryDimExpression<IndexType,Self,DivOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Self,MulOp<IndexType>>
	operator*(const IndexType& n) {
		return UnaryDimExpression<IndexType,Self,MulOp<IndexType>>(*this, n);
	};
	inline friend UnaryDimExpression<IndexType,Self,MulOp<IndexType>>
	operator*(const IndexType& n, Self& dims) {
		return UnaryDimExpression<IndexType,Self,MulOp<IndexType>>(dims, n);
	};
	inline friend std::ostream& operator<<(std::ostream& os, const Self& dims) {
		return os << dims.to_string() << std::endl;
	};
};

template<typename IndexType, typename LhsExpr, typename OpType>
class UnaryDimExpression :
		public DimExpression<UnaryDimExpression<IndexType,LhsExpr,OpType>,IndexType> {
public:
	inline UnaryDimExpression(LhsExpr& lhs, const IndexType& rhs) :
			lhs(lhs),
			rhs(rhs) { };
	inline IndexType operator()(size_t i) const {
		switch (i) {
			case 0:
				return OpType::apply(lhs.get_height(), rhs);
			case 1:
				return OpType::apply(lhs.get_width(), rhs);
			case 2:
				return OpType::apply(lhs.get_depth(), rhs);
			default:
				throw std::invalid_argument("illegal index value: " + i);
		}
	};
	inline IndexType get_height() const {
		return OpType::apply(lhs.get_height(), rhs);
	};
	inline IndexType get_width() const {
		return OpType::apply(lhs.get_width(), rhs);
	};
	inline IndexType get_depth() const {
		return OpType::apply(lhs.get_depth(), rhs);
	};
private:
	const LhsExpr& lhs;
	IndexType rhs;
};

template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDimExpression :
		public DimExpression<BinaryDimExpression<IndexType,LhsExpr,RhsExpr,OpType>,IndexType> {
public:
	inline BinaryDimExpression(const LhsExpr& lhs, const RhsExpr& rhs) :
			lhs(lhs),
			rhs(rhs) { };
	inline IndexType operator()(size_t i) const {
		switch (i) {
			case 0:
				return OpType::apply(lhs.get_height(), rhs.get_height());
			case 1:
				return OpType::apply(lhs.get_width(), rhs.get_width());
			case 2:
				return OpType::apply(lhs.get_depth(), rhs.get_depth());
			default:
				throw std::invalid_argument("illegal index value: " + i);
		}
	};
	inline IndexType get_height() const {
		return OpType::apply(lhs.get_height(), rhs.get_height());
	};
	inline IndexType get_width() const {
		return OpType::apply(lhs.get_width(), rhs.get_width());
	};
	inline IndexType get_depth() const {
		return OpType::apply(lhs.get_depth(), rhs.get_depth());
	};
protected:
	const LhsExpr& lhs;
	const RhsExpr& rhs;
};

/**
 * This only delays the evaluation and temporary storage of depth.
 */
template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDepthWiseDimExpression :
		public DimExpression<BinaryDepthWiseDimExpression<IndexType,LhsExpr,RhsExpr,OpType>,IndexType> {
public:
	inline BinaryDepthWiseDimExpression(const LhsExpr& lhs, const RhsExpr& rhs) :
			lhs(lhs),
			rhs(rhs),
			height(lhs.get_height()),
			width(lhs.get_width()) {
		assert(height == rhs.get_height() && width == rhs.get_width());
	};
	inline IndexType operator()(size_t i) const {
		switch (i) {
			case 0:
				return height;
			case 1:
				return width;
			case 2:
				return OpType::apply(lhs.get_depth(), rhs.get_depth());
			default:
				throw std::invalid_argument("illegal index value: " + i);
		}
	};
	inline IndexType get_height() const {
		return height;
	};
	inline IndexType get_width() const {
		return width;
	};
	inline IndexType get_depth() const {
		return OpType::apply(lhs.get_depth(), rhs.get_depth());
	};
protected:
	const LhsExpr& lhs;
	const RhsExpr& rhs;
	IndexType height;
	IndexType width;
};

template<typename IndexType>
class Dimensions : public DimExpression<Dimensions<IndexType>,IndexType> {
public:
	Dimensions(IndexType height, IndexType width, IndexType depth) :
			height(height),
			width(width),
			depth(depth) {
		assert(height > 0 && width > 0 && depth > 0);
	};
	Dimensions(IndexType height, IndexType width) :
			Dimensions(height, width, 1) { };
	Dimensions(IndexType height) :
			Dimensions(height, 1) { };
	Dimensions() :
			Dimensions(1) { };
	template<typename OtherDerived>
	Dimensions(const DimExpression<OtherDerived,IndexType>& dims) :
			Dimensions(dims.get_height(), dims.get_width(), dims.get_depth()) { };
	inline IndexType operator()(size_t i) const {
		switch (i) {
			case 0:
				return height;
			case 1:
				return width;
			case 2:
				return depth;
			default:
				throw std::invalid_argument("illegal index value: " + i);
		}
	};
	inline IndexType get_height() const {
		return height;
	};
	inline IndexType get_width() const {
		return width;
	};
	inline IndexType get_depth() const {
		return depth;
	};
private:
	IndexType height;
	IndexType width;
	IndexType depth;
};

} /* namespace cattle */

#endif /* DIMENSIONS_H_ */
