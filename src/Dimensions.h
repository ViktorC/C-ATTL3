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
class UnaryDimensionsExpression;
template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDimensionsExpression;
template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDepthWiseDimensionsExpression;

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
class DimensionsExpression {
	typedef DimensionsExpression<Derived,IndexType> Self;
	template<typename OtherDerived> using Other = DimensionsExpression<OtherDerived,IndexType>;
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
	inline BinaryDepthWiseDimensionsExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>
	add_along_depth(const Other<OtherDerived>& dims) {
		return BinaryDepthWiseDimensionsExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDepthWiseDimensionsExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>
	subtract_along_depth(const Other<OtherDerived>& dims) {
		return BinaryDepthWiseDimensionsExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDimensionsExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>
	operator+(const Other<OtherDerived>& dims) {
		return BinaryDimensionsExpression<IndexType,Self,Other<OtherDerived>,SumOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDimensionsExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>
	operator-(const Other<OtherDerived>& dims) {
		return BinaryDimensionsExpression<IndexType,Self,Other<OtherDerived>,SubOp<IndexType>>(*this, dims);
	};
	inline UnaryDimensionsExpression<IndexType,Self,SumOp<IndexType>>
	operator+(const IndexType& n) {
		return UnaryDimensionsExpression<IndexType,Self,SumOp<IndexType>>(*this, n);
	};
	inline UnaryDimensionsExpression<IndexType,Self,SubOp<IndexType>>
	operator-(const IndexType& n) {
		return UnaryDimensionsExpression<IndexType,Self,SubOp<IndexType>>(*this, n);
	};
	inline UnaryDimensionsExpression<IndexType,Self,DivOp<IndexType>>
	operator/(const IndexType& n) {
		return UnaryDimensionsExpression<IndexType,Self,DivOp<IndexType>>(*this, n);
	};
	inline UnaryDimensionsExpression<IndexType,Self,MulOp<IndexType>>
	operator*(const IndexType& n) {
		return UnaryDimensionsExpression<IndexType,Self,MulOp<IndexType>>(*this, n);
	};
	inline friend UnaryDimensionsExpression<IndexType,Self,MulOp<IndexType>>
	operator*(const IndexType& n, Self& dims) {
		return UnaryDimensionsExpression<IndexType,Self,MulOp<IndexType>>(dims, n);
	};
	inline friend std::ostream& operator<<(std::ostream& os, const Self& dims) {
		return os << dims.to_string() << std::endl;
	};
};

template<typename IndexType, typename LhsExpr, typename OpType>
class UnaryDimensionsExpression :
		public DimensionsExpression<UnaryDimensionsExpression<IndexType,LhsExpr,OpType>,IndexType> {
public:
	inline UnaryDimensionsExpression(LhsExpr& lhs, const IndexType& rhs) :
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
class BinaryDimensionsExpression :
		public DimensionsExpression<BinaryDimensionsExpression<IndexType,LhsExpr,RhsExpr,OpType>,IndexType> {
public:
	inline BinaryDimensionsExpression(const LhsExpr& lhs, const RhsExpr& rhs) :
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

template<typename IndexType, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDepthWiseDimensionsExpression :
		public DimensionsExpression<BinaryDepthWiseDimensionsExpression<IndexType,LhsExpr,RhsExpr,OpType>,IndexType> {
public:
	inline BinaryDepthWiseDimensionsExpression(const LhsExpr& lhs, const RhsExpr& rhs) :
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
class Dimensions : public DimensionsExpression<Dimensions<IndexType>,IndexType> {
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
	Dimensions(const DimensionsExpression<OtherDerived,IndexType>& dims) :
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
