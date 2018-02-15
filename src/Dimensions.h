/*
 * Dimensions.h
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef DIMENSIONS_H_
#define DIMENSIONS_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cattle {

/**
 * NOTE: Expression templates are unwarranted for this class as its objects are very unlikely to
 * be used in complex expressions. They have only been implemented for the sake of practical learning.
 */

// Forward declarations.
template<typename IndexType, size_t Rank, typename LhsExpr, typename OpType>
class UnaryDimExpression;
template<typename IndexType, size_t Rank, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDimExpression;
template<typename IndexType, size_t Rank, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryRankWiseDimExpression;

// Arithmetic operations.
template<typename Operand> class SumOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs + rhs; };
};
template<typename Operand> class SubOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs - rhs; };
};
template<typename Operand> class MulOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs * rhs; };
};
template<typename Operand> class DivOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs / rhs; };
};

template<typename Derived, typename IndexType, size_t Rank>
class DimExpression {
	static_assert(Rank > 0, "illegal rank");
	typedef DimExpression<Derived,IndexType,Rank> Self;
	template<typename OtherDerived> using Other = DimExpression<OtherDerived,IndexType,Rank>;
public:
	inline operator Derived&() {
		return static_cast<Derived&>(*this);
	};
	inline operator const Derived&() const {
		return static_cast<const Derived&>(*this);
	};
	inline IndexType operator()(size_t i) const {
		return static_cast<const Derived&>(*this)(i);
	};
	inline IndexType get_volume() const {
		int volume = 1;
		for (size_t i = 0; i < Rank; i++)
			volume *= (*this)(i);
		return volume;
	};
	inline std::string to_string() const {
		std::stringstream strm;
		strm << "[" + std::to_string((*this)(0));
		for (size_t i = 1; i < Rank; i++)
			strm << "," << std::to_string((*this)(i));
		strm << "]";
		return strm.str();
	};
	template<typename OtherDerived>
	inline BinaryRankWiseDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SumOp<IndexType>>
	add_along_rank(const Other<OtherDerived>& dims, size_t rank) const {
		return BinaryRankWiseDimExpression<IndexType,Rank,Self,
				Other<OtherDerived>,SumOp<IndexType>>(*this, dims, rank);
	};
	template<typename OtherDerived>
	inline BinaryRankWiseDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SubOp<IndexType>>
	subtract_along_rank(const Other<OtherDerived>& dims, size_t rank) const {
		return BinaryRankWiseDimExpression<IndexType,Rank,Self,
				Other<OtherDerived>,SubOp<IndexType>>(*this, dims, rank);
	};
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SumOp<IndexType>>
	operator+(const Other<OtherDerived>& dims) const {
		return BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SumOp<IndexType>>(*this, dims);
	};
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SubOp<IndexType>>
	operator-(const Other<OtherDerived>& dims) const {
		return BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SubOp<IndexType>>(*this, dims);
	};
	inline UnaryDimExpression<IndexType,Rank,Self,SumOp<IndexType>>
	operator+(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,SumOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Rank,Self,SubOp<IndexType>>
	operator-(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,SubOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Rank,Self,DivOp<IndexType>>
	operator/(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,DivOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>
	operator*(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>(*this, n);
	};
	inline friend UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>
	operator*(const IndexType& n, const Self& dims) {
		return UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>(dims, n);
	};
	template<typename OtherDerived, typename OtherIndexType, size_t OtherRank>
	inline bool operator==(const DimExpression<OtherDerived,OtherIndexType,OtherRank>& dims) const {
		return false;
	};
	template<typename OtherDerived>
	inline bool operator==(const Other<OtherDerived>& dims) const {
		bool equal = true;
		for (size_t i = 0; i < Rank; i++) {
			if ((*this)(i) != dims(i))
				equal = false;
		}
		return equal;
	};
	inline friend std::ostream& operator<<(std::ostream& os, const Self& dims) {
		return os << dims.to_string();
	};
};

template<typename IndexType, size_t Rank, typename LhsExpr, typename OpType>
class UnaryDimExpression :
		public DimExpression<UnaryDimExpression<IndexType,Rank,LhsExpr,OpType>,IndexType,Rank> {
public:
	inline UnaryDimExpression(const LhsExpr& lhs, const IndexType& rhs) :
			lhs(lhs),
			rhs(rhs) { };
	inline IndexType operator()(size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + i);
		return OpType::apply(lhs(i), rhs);
	};
private:
	const LhsExpr& lhs;
	IndexType rhs;
};

template<typename IndexType, size_t Rank, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDimExpression :
		public DimExpression<BinaryDimExpression<IndexType,Rank,LhsExpr,RhsExpr,OpType>,IndexType,Rank> {
public:
	inline BinaryDimExpression(const LhsExpr& lhs, const RhsExpr& rhs) :
			lhs(lhs),
			rhs(rhs) { };
	inline IndexType operator()(size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + i);
		return OpType::apply(lhs(i), rhs(i));
	};
protected:
	const LhsExpr& lhs;
	const RhsExpr& rhs;
};

template<typename IndexType, size_t Rank, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryRankWiseDimExpression :
		public DimExpression<BinaryRankWiseDimExpression<IndexType,Rank,LhsExpr,RhsExpr,OpType>,IndexType,Rank> {
public:
	inline BinaryRankWiseDimExpression(const LhsExpr& lhs, const RhsExpr& rhs, size_t rank) :
			lhs(lhs),
			rhs(rhs),
			rank(rank) {
		assert(rank < Rank);
	};
	inline IndexType operator()(size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + i);
		return i == rank ? OpType::apply(lhs(i), rhs(i)) : lhs(i);
	};
protected:
	const LhsExpr& lhs;
	const RhsExpr& rhs;
	size_t rank;
};

template<typename IndexType, size_t Rank>
class Dimensions : public DimExpression<Dimensions<IndexType,Rank>,IndexType,Rank> {
	friend class Dimensions<IndexType,Rank - 1>;
	friend class Dimensions<IndexType,Rank + 1>;
public:
	Dimensions() :
			values(Rank, 1) { };
	Dimensions(const std::initializer_list<IndexType>& values) :
			Dimensions() {
		assert(values.size() <= Rank);
		std::copy(values.begin(), values.end(), this->values.begin());
	};
	Dimensions(const std::array<IndexType,Rank>& array) :
			Dimensions() {
		assert(array.size() <= Rank);
		std::copy(array.begin(), array.end(), values.begin());
	};
	template<typename OtherDerived>
	Dimensions(const DimExpression<OtherDerived,IndexType,Rank>& dims) :
			Dimensions() {
		for (size_t i = 0; i < Rank; i++)
			values[i] = dims(i);
	};
	template<size_t Ranks = 1>
	inline Dimensions<IndexType,Rank + Ranks> promote() const {
		Dimensions<IndexType,Rank + Ranks> promoted;
		std::copy(values.begin(), values.end(), promoted.values.begin() + Ranks);
		return promoted;
	};
	template<size_t Ranks = 1>
	inline Dimensions<IndexType,Rank - Ranks> demote() const {
		static_assert(Rank > Ranks, "rank must be greater than the number of ranks to demote by");
		Dimensions<IndexType,Rank - Ranks> demoted;
		std::copy(values.begin() + Ranks, values.end(), demoted.values.begin());
		return demoted;
	};
	inline IndexType operator()(size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + i);
		return values[i];
	};
	inline IndexType& operator()(size_t i) {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + i);
		return values[i];
	};
	inline operator std::array<IndexType,Rank>() const {
		std::array<IndexType,Rank> array;
		std::copy(values.begin(), values.end(), array.begin());
		return array;
	};
private:
	std::vector<IndexType> values;
};

} /* namespace cattle */

#endif /* DIMENSIONS_H_ */
