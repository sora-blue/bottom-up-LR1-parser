# resolve issue of class declaration order after Python 3.7
from __future__ import annotations

from typing import Sized

from .constants import *


# 终结符号和非终结符号
class Symbol:
    def __init__(self, sType: int, name: str, content: str = None):
        self.sType = sType
        self.name = name
        self.content = name if content is None else content

    def __str__(self):
        return self.content

    def __hash__(self):
        return hash(str(self.sType) + self.name)

    def __eq__(self, other: Symbol):
        if self.sType != other.sType:
            return False
        return self.name == other.name

    @staticmethod
    def fromStr(content: str, end_symbols: list[str]):
        if content in end_symbols:
            return Symbol(CONSTANT_GENERATOR_TYPE_END, content)
        return Symbol(CONSTANT_GENERATOR_TYPE_NONE_END, content)


# 产生式
class Generator:
    def __init__(self, start: Symbol, end: list[Symbol], func: callable or None = None):
        self.start = start
        self.end = end
        self.func = func

    def __str__(self):
        return f'{self.start.name} -> {" ".join([s.name for s in self.end])}'


# 多个产生式后部放在一块，从朴素写法 转换成 产生式的数组
class ComposedGenerator:
    def __init__(self, start: str, end: list[list[str]], func: callable or None = None):
        self.start = start
        self.end = end
        self.func = func
        pass

    def toListOfGenerator(self, end_symbols: list[str]):
        assert len(self.end) > 0 and type(self.end[0]) == list
        ret = []
        retStart = Symbol.fromStr(self.start, end_symbols)
        for g in self.end:
            ret.append(Generator(start=retStart, end=[Symbol.fromStr(s, end_symbols) for s in g], func=self.func))
        return ret


# 项
class Item:
    def __init__(self, start: Symbol, end: list[Symbol], point: int, lf_symbol: Symbol, func: callable or None = None):
        self.start = start
        self.end = end
        self.point = point
        self.lf_symbol = lf_symbol
        self.func = func

    def __hash__(self):
        r1 = hash(self.start)
        r2 = hash("".join([str(s) for s in self.end]))
        r3 = hash(self.point)
        r4 = hash(self.lf_symbol)
        return r1 + r2 + r3 + r4

    # 用于set中比较
    def __eq__(self, other: Item):
        if self.start != other.start:
            return False
        if self.end != other.end:
            return False
        if self.lf_symbol != other.lf_symbol:
            return False
        return self.point == other.point

    # 返回可读的形式
    def __str__(self):
        retStr = f'[{self.start} -> '
        pointStr = '· '
        lfStr = f'| {self.lf_symbol}]'
        if len(self.end) == 1 and self.end[0].name == CONSTANT_GENERATOR_NONE:
            return f'{retStr}{pointStr}{lfStr}'
        cnt = self.point
        for symbol in self.end:
            if cnt == 0:
                retStr += pointStr
            cnt -= 1
            retStr += f'{symbol.name} '
        if cnt == 0:
            retStr += pointStr
        retStr += lfStr
        return retStr


# 项集
class ItemSet(Sized):
    def __len__(self) -> int:
        return len(self._items)

    def __init__(self, itemList: list[Item] or set[Item]):
        self._items = dict(zip(itemList, [None for _ in range(len(itemList))]))
        self._cached_hash_val = None

    def __iter__(self):
        return self._items.keys().__iter__()

    def __hash__(self):
        if self._cached_hash_val is not None:
            return self._cached_hash_val
        val = 0
        for item in self._items.keys():
            val += hash(item)
        val = int(val)
        self._cached_hash_value = val
        return val

    # 判断是否相等，对使用set很重要
    def __eq__(self, other):
        if hash(self) != hash(other):
            return False
        for item in self:
            if item not in other:
                return False
        for item in other:
            if item not in self:
                return False
        return True

    def __str__(self):
        retStr = SPLIT_LINE + '\n'
        for item in reversed([i for i in self]):
            retStr += f'{item}\n'
        return retStr

    def add(self, item: Item):
        self._items.setdefault(item, None)

    def union(self, other: ItemSet):
        newItemSet = other
        for item in self._items.keys():
            newItemSet._items.setdefault(item)
        return newItemSet

    def toList(self):
        return [item for item in self._items.keys()]


# CFG 文法
class GrammarCFG:
    def __init__(self, start_mark: str, end_mark: str,
                 end_symbols: list[str] or set[str], c_generators: list[ComposedGenerator] or list[Generator]):
        self.start_mark = Symbol(CONSTANT_GENERATOR_TYPE_NONE_END, start_mark)
        self.end_mark = Symbol(CONSTANT_GENERATOR_TYPE_END, end_mark)  # not tested
        self.end_symbols = [Symbol(CONSTANT_GENERATOR_TYPE_END, s) for s in end_symbols]
        self.generators = []
        if isinstance(c_generators[0], ComposedGenerator):
            for c_g in c_generators:
                self.generators += c_g.toListOfGenerator(end_symbols)
        elif isinstance(c_generators[0], Generator):
            self.generators = c_generators
        # 由非终结符号索引产生式
        self.map_generators = {}
        for g in self.generators:
            self.map_generators.setdefault(g.start, [])
            self.map_generators[g.start].append(g)
