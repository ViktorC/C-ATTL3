/*
 * Dimensions.hpp
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_DIMENSIONS_H_
#define CATTL3_DIMENSIONS_H_

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
 * NOTE: Expression templates are rather unwarranted for this class as its objects are unlikely to
 * be used in complex expressions. They have only been implemented for the sake of practical learning.
 */

template<typename Derived, typename IndexType, std::size_t Rank>
class DimExpression;

/**
 * A class representing dimensions along one or more ranks. It can describe the dimensionality of
 * tensors of arbitrary ranks.
 */
template<typename IndexType, std::size_t Rank>
class Dimensions : public DimExpression<Dimensions<IndexType,Rank>,IndexType,Rank> {
	template<typename OtherIndexType, std::size_t OtherRank>
	friend class Dimensions;
public:
	inline Dimensions(const std::array<IndexType,Rank>& array) :
			values(Rank) {
		std::copy(array.begin(), array.end(), values.begin());
	}
	inline Dimensions(const std::initializer_list<IndexType>& values) :
			values(Rank, 1) {
		assert(values.size() <= Rank);
		std::copy(values.begin(), values.end(), this->values.begin());
	}
	inline Dimensions(IndexType values...) :
			Dimensions({ values }) { }
	template<typename OtherDerived>
	inline Dimensions(const DimExpression<OtherDerived,IndexType,Rank>& dims) :
			Dimensions() {
		for (std::size_t i = 0; i < Rank; ++i)
			values[i] = dims(i);
	}
	/**
	 * A constant method that returns a copy of the instance with n additional ranks prepended to it.
	 *
	 * @return A new Dimensions instance with additional n ranks of size 1 prepended to it.
	 */
	template<std::size_t Ranks = 1>
	inline Dimensions<IndexType,Rank + Ranks> promote() const {
		Dimensions<IndexType,Rank + Ranks> promoted;
		std::copy(values.begin(), values.end(), promoted.values.begin() + Ranks);
		return promoted;
	}
	/**
	 * A constant method that returns a copy of the instance without the n most-significant ranks.
	 *
	 * @return A new Dimensions instance with the first n ranks removed.
	 */
	template<std::size_t Ranks = 1>
	inline Dimensions<IndexType,Rank - Ranks> demote() const {
		static_assert(Rank > Ranks, "rank must be greater than the number of ranks to demote by");
		Dimensions<IndexType,Rank - Ranks> demoted;
		std::copy(values.begin() + Ranks, values.end(), demoted.values.begin());
		return demoted;
	}
	/**
	 * A constant method that returns a copy of the instance with n ranks appended to it.
	 *
	 * @return A new Dimensions instance with additional n ranks with size 1 appended to it.
	 */
	template<std::size_t Ranks = 1>
	inline Dimensions<IndexType,Rank + Ranks> extend() const {
		Dimensions<IndexType,Rank + Ranks> extended;
		std::copy(values.begin(), values.end(), extended.values.begin());
		return extended;
	}
	/**
	 * A constant method that returns a copy of the instance without the n least-significant ranks.
	 *
	 * @return A new Dimensions instance with the last n ranks removed.
	 */
	template<std::size_t Ranks = 1>
	inline Dimensions<IndexType,Rank - Ranks> contract() const {
		static_assert(Rank > Ranks, "rank must be greater than the number of ranks to contract by");
		Dimensions<IndexType,Rank - Ranks> contracted;
		std::copy(values.begin(), values.end() - Ranks, contracted.values.begin());
		return contracted;
	}
	/**
	 * A simple constant method that returns a copy of the numeral representing the
	 * number of dimensions along a given rank.
	 *
	 * @param i The index of the rank whose dimensionality is to be returned.
	 * @return The dimensionality of the i-th rank.
	 */
	inline IndexType operator()(std::size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + std::to_string(i));
		return values[i];
	}
	/**
	 * A that returns a non-constant reference to the numeral representing the
	 * number of dimensions along a given rank.
	 *
	 * @param i The index of the rank whose dimensionality is to be returned.
	 * @return The dimensionality of the i-th rank.
	 */
	inline IndexType& operator()(std::size_t i) {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + std::to_string(i));
		return values[i];
	}
	/**
	 * A constant conversion operator that returns an array with the contents
	 * of the instance.
	 */
	inline operator std::array<IndexType,Rank>() const {
		std::array<IndexType,Rank> array;
		std::copy(values.begin(), values.end(), array.begin());
		return array;
	}
private:
	std::vector<IndexType> values;
};

/**
 * An expression representing an operation between a numeral and all ranks of a dimension expression.
 */
template<typename IndexType, std::size_t Rank, typename LhsExpr, typename OpType>
class UnaryDimExpression :
		public DimExpression<UnaryDimExpression<IndexType,Rank,LhsExpr,OpType>,IndexType,Rank> {
public:
	inline UnaryDimExpression(const LhsExpr& lhs, const IndexType& rhs) :
			lhs(lhs),
			rhs(rhs) { };
	inline IndexType operator()(std::size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + std::to_string(i));
		return OpType::apply(lhs(i), rhs);
	}
private:
	const LhsExpr& lhs;
	IndexType rhs;
};

/**
 * An expression representing an operation between a numeral and a single rank of a dimension expression.
 */
template<typename IndexType, std::size_t Rank, typename LhsExpr, typename OpType>
class UnaryRankWiseDimExpression :
		public DimExpression<UnaryRankWiseDimExpression<IndexType,Rank,LhsExpr,OpType>,IndexType,Rank> {
public:
	inline UnaryRankWiseDimExpression(const LhsExpr& lhs, const IndexType& rhs, std::size_t rank) :
			lhs(lhs),
			rhs(rhs),
			rank(rank) {
		assert(rank < Rank);
	};
	inline IndexType operator()(std::size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + std::to_string(i));
		return i == rank ? OpType::apply(lhs(i), rhs) : lhs(i);
	}
private:
	const LhsExpr& lhs;
	IndexType rhs;
	std::size_t rank;
};

/**
 * An expression representing an operation between two dimension expressions of the same rank.
 */
template<typename IndexType, std::size_t Rank, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryDimExpression :
		public DimExpression<BinaryDimExpression<IndexType,Rank,LhsExpr,RhsExpr,OpType>,IndexType,Rank> {
public:
	inline BinaryDimExpression(const LhsExpr& lhs, const RhsExpr& rhs) :
			lhs(lhs),
			rhs(rhs) { };
	inline IndexType operator()(std::size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + std::to_string(i));
		return OpType::apply(lhs(i), rhs(i));
	}
protected:
	const LhsExpr& lhs;
	const RhsExpr& rhs;
};

/**
 * An expression representing an operation along a single rank of two dimension expressions of the
 * same rank.
 */
template<typename IndexType, std::size_t Rank, typename LhsExpr, typename RhsExpr, typename OpType>
class BinaryRankWiseDimExpression :
		public DimExpression<BinaryRankWiseDimExpression<IndexType,Rank,LhsExpr,RhsExpr,OpType>,IndexType,Rank> {
public:
	inline BinaryRankWiseDimExpression(const LhsExpr& lhs, const RhsExpr& rhs, std::size_t rank) :
			lhs(lhs),
			rhs(rhs),
			rank(rank) {
		assert(rank < Rank);
	}
	inline IndexType operator()(std::size_t i) const {
		if (i < 0 || i >= Rank)
			throw std::invalid_argument("illegal index value: " + std::to_string(i));
		return i == rank ? OpType::apply(lhs(i), rhs(i)) : lhs(i);
	}
protected:
	const LhsExpr& lhs;
	const RhsExpr& rhs;
	std::size_t rank;
};

// Arithmetic operations.
template<typename Operand> class SumOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs + rhs; }
};
template<typename Operand> class SubOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs - rhs; }
};
template<typename Operand> class MulOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs * rhs; }
};
template<typename Operand> class DivOp {
public: inline static Operand apply(Operand lhs, Operand rhs) { return lhs / rhs; }
};

/**
 * The base class of all dimension expressions.
 */
template<typename Derived, typename IndexType, std::size_t Rank>
class DimExpression {
	static_assert(Rank > 0, "illegal rank");
	typedef DimExpression<Derived,IndexType,Rank> Self;
	template<typename OtherDerived> using Other = DimExpression<OtherDerived,IndexType,Rank>;
public:
	inline operator Derived&() {
		return static_cast<Derived&>(*this);
	}
	inline operator const Derived&() const {
		return static_cast<const Derived&>(*this);
	}
	/**
	 * Evaluates the expression along the given rank.
	 *
	 * @param i The index of the single rank along which the expression is to be evaluated.
	 * @return The result of the expression evaluated along the i-th rank.
	 */
	inline IndexType operator()(std::size_t i) const {
		return static_cast<const Derived&>(*this)(i);
	}
	/**
	 * A constant method the returns the volume of the expression.
	 *
	 * @return The product of the dimensions of each rank of the instance.
	 */
	inline IndexType get_volume() const {
		int volume = 1;
		for (std::size_t i = 0; i < Rank; ++i)
			volume *= (*this)(i);
		return volume;
	}
	/**
	 * A method for forcing the evaluation of the expression.
	 *
	 * @return A Dimensions instance containing the results of the evaluated
	 * expression.
	 */
	inline Dimensions<IndexType,Rank> eval() {
		return Dimensions<IndexType,Rank>(*this);
	}
	/**
	 * It evaluates the expression and returns a string containing the results.
	 *
	 * @return A string representation of the evaluated expression.
	 */
	inline std::string to_string() const {
		std::stringstream strm;
		strm << "[" + std::to_string((*this)(0));
		for (std::size_t i = 1; i < Rank; ++i)
			strm << "," << std::to_string((*this)(i));
		strm << "]";
		return strm.str();
	}
	/**
	 * @param n The value to add.
	 * @param rank The rank to which the value is to be added.
	 * @return An expression representing the addition of the specified value to the
	 * specified rank of the dimension expression.
	 */
	inline UnaryRankWiseDimExpression<IndexType,Rank,Self,SumOp<IndexType>>
	add_along_rank(const IndexType& n, std::size_t rank) const {
		return UnaryRankWiseDimExpression<IndexType,Rank,Self,SumOp<IndexType>>(*this, n, rank);
	}
	/**
	 * @param n The value to subtract.
	 * @param rank The rank from which the value is to be subtracted.
	 * @return An expression representing the subtraction of the specified value from the
	 * specified rank of the dimension expression.
	 */
	inline UnaryRankWiseDimExpression<IndexType,Rank,Self,SubOp<IndexType>>
	subtract_along_rank(const IndexType& n, std::size_t rank) const {
		return UnaryRankWiseDimExpression<IndexType,Rank,Self,SubOp<IndexType>>(*this, n, rank);
	}
	/**
	 * @param n The value to multiply by.
	 * @param rank The rank which is to be multiplied by the value.
	 * @return An expression representing the multiplication of the specified rank of the
	 * dimension expression by the specified value.
	 */
	inline UnaryRankWiseDimExpression<IndexType,Rank,Self,MulOp<IndexType>>
	multiply_along_rank(const IndexType& n, std::size_t rank) const {
		return UnaryRankWiseDimExpression<IndexType,Rank,Self,MulOp<IndexType>>(*this, n, rank);
	}
	/**
	 * @param n The value to divide by.
	 * @param rank The rank which is to be divide by the value.
	 * @return An expression representing the division of the specified rank of the
	 * dimension expression by the specified value.
	 */
	inline UnaryRankWiseDimExpression<IndexType,Rank,Self,DivOp<IndexType>>
	divide_along_rank(const IndexType& n, std::size_t rank) const {
		return UnaryRankWiseDimExpression<IndexType,Rank,Self,DivOp<IndexType>>(*this, n, rank);
	}
	/**
	 * @param dims The dimension expression to add.
	 * @param rank The rank along which the expressions are to be added.
	 * @return An expression representing the addition of the specified expression
	 * dimension to the instance on which the method is invoked along the specified rank.
	 */
	template<typename OtherDerived>
	inline BinaryRankWiseDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SumOp<IndexType>>
	add_along_rank(const Other<OtherDerived>& dims, std::size_t rank) const {
		return BinaryRankWiseDimExpression<IndexType,Rank,Self,
				Other<OtherDerived>,SumOp<IndexType>>(*this, dims, rank);
	}
	/**
	 * @param dims The dimension expression to subtract.
	 * @param rank The rank along which the dimension expression is to be subtracted from
	 * the instance on which the method is invoked.
	 * @return An expression representing the subtraction of the specified expression
	 * dimension from the instance on which the method is invoked along the specified rank.
	 */
	template<typename OtherDerived>
	inline BinaryRankWiseDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SubOp<IndexType>>
	subtract_along_rank(const Other<OtherDerived>& dims, std::size_t rank) const {
		return BinaryRankWiseDimExpression<IndexType,Rank,Self,
				Other<OtherDerived>,SubOp<IndexType>>(*this, dims, rank);
	}
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SumOp<IndexType>>
	operator+(const Other<OtherDerived>& dims) const {
		return BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SumOp<IndexType>>(*this, dims);
	}
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SubOp<IndexType>>
	operator-(const Other<OtherDerived>& dims) const {
		return BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,SubOp<IndexType>>(*this, dims);
	}
	template<typename OtherDerived>
	inline BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,MulOp<IndexType>>
	operator*(const Other<OtherDerived>& dims) const {
		return BinaryDimExpression<IndexType,Rank,Self,Other<OtherDerived>,MulOp<IndexType>>(*this, dims);
	}
	inline UnaryDimExpression<IndexType,Rank,Self,SumOp<IndexType>>
	operator+(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,SumOp<IndexType>>(*this, n);
	};
	inline UnaryDimExpression<IndexType,Rank,Self,SubOp<IndexType>>
	operator-(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,SubOp<IndexType>>(*this, n);
	}
	inline UnaryDimExpression<IndexType,Rank,Self,DivOp<IndexType>>
	operator/(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,DivOp<IndexType>>(*this, n);
	}
	inline UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>
	operator*(const IndexType& n) const {
		return UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>(*this, n);
	}
	inline friend UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>
	operator*(const IndexType& n, const Self& dims) {
		return UnaryDimExpression<IndexType,Rank,Self,MulOp<IndexType>>(dims, n);
	};
	template<typename OtherDerived, typename OtherIndexType, std::size_t OtherRank>
	inline bool operator==(const DimExpression<OtherDerived,OtherIndexType,OtherRank>& dims) const {
		return false;
	}
	template<typename OtherDerived>
	inline bool operator==(const Other<OtherDerived>& dims) const {
		for (std::size_t i = 0; i < Rank; ++i) {
			if ((*this)(i) != dims(i))
				return false;
		}
		return true;
	}
	inline bool operator==(const std::array<IndexType,Rank>& dims) const {
		for (std::size_t i = 0; i < Rank; ++i) {
			if ((*this)(i) != dims[i])
				return false;
		}
		return true;
	}
	inline friend std::ostream& operator<<(std::ostream& os, const Self& dims) {
		return os << dims.to_string();
	}
};

} /* namespace cattle */

#endif /* CATTL3_DIMENSIONS_H_ */
