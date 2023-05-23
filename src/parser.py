from __future__ import annotations

import collections

from .grammar import *


# ACTION的动作
class LRSheetAction:
    def __init__(self, action_type, action_to=None):
        self.action_type = action_type
        self.action_to = action_to

    def __eq__(self, other: LRSheetAction):
        if self.action_type != other.action_type:
            return False
        return self.action_to == other.action_to


# 格局
class Configuration:
    def __init__(self, parse_stk: list[int], cur_symbols: list[Symbol],
                 input_symbols: list[Symbol], lr_sheet: LRSheet):
        self.parse_stk = parse_stk
        self.cur_symbols = cur_symbols
        self.input_symbols = input_symbols
        self.lr_sheet = lr_sheet
        self._cnt = 0
        self.__name__ = "Parser config"

    def show_current(self, action):
        print(f'---step {self._cnt}---')
        print('栈：')
        p_line = " ".join([str(i) for i in self.parse_stk])
        print(p_line)

        print('符号：')
        p_line = " ".join([str(s) for s in self.cur_symbols])
        print(p_line)

        print('输入：')
        p_line = " ".join([str(s) for s in self.input_symbols])
        print(p_line)

        print('动作：')
        p_line = str(action.action_type)
        print(p_line)

        print('SDD：')
        for s in self.cur_symbols:
            print(s, s.__dict__)

    # 返回一个动作，规约时还返回一个携带语法制导动作的项和栈中的符号
    def _forward(self) -> [LRSheetAction, [Item, list[Symbol]] or None]:
        action = self.lr_sheet.action(self.parse_stk[-1], self.input_symbols[0])
        if action.action_type == CONSTANT_ACTION_FORWARD_J:
            self.cur_symbols.append(self.input_symbols.pop(0))
            self.parse_stk.append(action.action_to)
        elif action.action_type == CONSTANT_ACTION_REDUCTION_A:
            item = action.action_to[1]
            len_end = len(item.end)
            item_start = action.action_to[0]
            symbols = self.cur_symbols[-len_end:]
            self.parse_stk = self.parse_stk[:-len_end]
            self.parse_stk.append(self.lr_sheet.goto(self.parse_stk[-1], item_start))
            self.cur_symbols = self.cur_symbols[:-len_end]
            self.cur_symbols.append(item_start)
            return [action, [item, symbols]]
        elif action.action_type == CONSTANT_ACTION_ACCEPT:
            return [action, None]
        elif action.action_type == CONSTANT_ACTION_NONE:
            if self.input_symbols[0].name == CONSTANT_GENERATOR_NONE:
                print(f'{self.__name__}: Wrong when action(state {self.parse_stk[-1]}, symbol {self.input_symbols[0]})')
                self.input_symbols.pop(0)
                tmp_action = self.lr_sheet.action(self.parse_stk[-1], self.input_symbols[0])
                while tmp_action.action_type == CONSTANT_ACTION_NONE and len(self.input_symbols) > 0:
                    s = self.input_symbols.pop(0)
                    print(f'{self.__name__}: Discard {s}')
                    tmp_action = self.lr_sheet.action(self.parse_stk[-1], self.input_symbols[0])
                if tmp_action.action_type == CONSTANT_ACTION_NONE and len(self.input_symbols) == 0:
                    raise ValueError(f'{self.__name__}: Not accepted')
                return self._forward()
            self.input_symbols.insert(0, Symbol(CONSTANT_GENERATOR_TYPE_END,
                                                CONSTANT_GENERATOR_NONE))
        return [action, None]

    def forward(self, show=False) -> [LRSheetAction, list or None]:
        self._cnt += 1
        action, l = self._forward()
        if show:
            self.show_current(action)
        if action.action_type == CONSTANT_ACTION_ACCEPT:
            # 为空
            if len(self.input_symbols) == 0 \
                    or (len(self.input_symbols) == 1 and self.input_symbols[0].content == '$'):
                return [action, l]
            raise ValueError("算法为接受时，输入不为空")
        return [action, l]


# LR1语法分析表
class LRSheet:
    def __init__(self, cfg: GrammarCFG, states: list[ItemSet]):
        self._cfg = cfg
        self._states = states
        self.state_idx = 0
        self.state = self._states[self.state_idx]
        self._map_action = {}
        self._map_goto = {}
        self._factory_key = collections.namedtuple('action', ['state', 'symbol'])

    def action(self, state_idx: int, symbol: Symbol) -> LRSheetAction:
        key = self._factory_key(state_idx, symbol)
        if key not in self._map_action:
            return LRSheetAction(CONSTANT_ACTION_NONE)
        return self._map_action[key]

    def set_action(self, state_idx: int, symbol: Symbol, action: LRSheetAction) -> bool:
        key = self._factory_key(state_idx, symbol)
        if key in self._map_action and self._map_action[key] != action:
            print(f'Wrong: state {state_idx} symbol {symbol}')
            raise ValueError("Not a LR(1) grammar!")
        self._map_action[key] = action
        return True

    def goto(self, state_idx: int, symbol: Symbol) -> int:
        key = self._factory_key(state_idx, symbol)
        if key not in self._map_goto:
            return -1
        return self._map_goto[key]

    def set_goto(self, state_idx: int, symbol: Symbol, dst: int) -> bool:
        key = self._factory_key(state_idx, symbol)
        if key in self._map_goto:
            print(f'Wrong: state {state_idx} symbol {symbol}')
            raise ValueError("Not a LR(1) grammar!")
        self._map_goto[key] = dst
        return True

    def set_initial_state(self, state_idx):
        self.state_idx = state_idx
        self.state = self._states[self.state_idx]


# 符号表
class SymbolTable:
    def __init__(self):
        self._factory_val = collections.namedtuple('entry', ['name', 'st_type'])
        self._table = dict()
        self.__name__ = "symbol table"

    def put_entry(self, name: str, st_type: str):
        if name in self._table:
            raise ValueError(f'{self.__name__}: {name} already exists as {self._table[name]}')
        val = self._factory_val(name, st_type)
        self._table[name] = val

    def get_entry_type(self, name: str):
        if name not in self._table:
            raise ValueError(f'{self.__name__}: {name} not in {self._table[name]}')
        return self._table[name].st_type

    def __str__(self):
        retStr = ''
        retStr += f'--- 符号表 ---\n'
        retStr += f'类型\t\t名称\n'
        retStr += f'{SPLIT_LINE}\n'
        for key in self._table.keys():
            retStr += f'{self._table[key].st_type}\t\t{key}\n'
        retStr = retStr[:-1]
        return retStr


class Parser:
    def __init__(self, cfg: GrammarCFG, debug_mode=False):
        self._cfg = cfg
        self._map_follow = {}
        self._map_first = {}
        self._debug_mode = debug_mode
        self._config = None
        self._sTable = SymbolTable()

    # 对LR(1)项集求闭包 P167
    def get_closure(self, I: ItemSet):
        new_items = I
        cur_items = ItemSet([])
        # 记录迭代中已经求过闭包的符号
        is_iterated = []
        # 不递归产生式左部
        # for item in I:
        #    if item.point == 0:
        #        is_iterated.add(item.start)
        # 如果有新的项需要求
        while len(new_items) > 0:
            tmp_items = ItemSet([])
            # 对于每个项
            for item in new_items:
                # 对于紧跟产生式中点后面的符号
                for i in range(item.point, min(len(item.end), item.point + 1)):
                    symbol = item.end[i]
                    f_list = item.end[i + 1:] + [item.lf_symbol]
                    # 终结符号
                    if symbol in self._cfg.end_symbols:
                        continue
                    # 已遍历的 非终结符号
                    if [symbol, f_list] in is_iterated:
                        continue
                    # 未识别的 非终结符号
                    if symbol not in self._cfg.map_generators.keys():
                        continue
                    # 将新的非终结符号对应的产生式加入项集
                    for g in self._cfg.map_generators[symbol]:
                        # 对于first(\beta a) 中的每个终结符号 b
                        f_s_list = self.get_chain_first(f_list)
                        for b in f_s_list:
                            # Note: 特判 结束标记
                            if b not in self._cfg.end_symbols and b != self._cfg.end_mark:
                                continue
                            new_item = Item(start=g.start, end=g.end, point=0, lf_symbol=b, func=g.func)
                            tmp_items.add(new_item)
                        # 记录已遍历的非终结符号和其f_list
                        is_iterated.append([symbol, f_list])
            cur_items = cur_items.union(new_items)
            new_items = tmp_items
        cur_items = cur_items.union(new_items)
        return cur_items

    # 对LR(1)项集求转移 P167
    def get_goto(self, I: ItemSet, X: Symbol):
        ret = ItemSet([])
        for item in I:
            # 已到达末尾
            if item.point >= len(item.end):
                continue
            # 后一个符号不是X
            if item.end[item.point] != X:
                continue
            tmp_item = Item(start=item.start, end=item.end, point=item.point + 1,
                            lf_symbol=item.lf_symbol, func=item.func)
            ret.add(tmp_item)
        ret = self.get_closure(ret)
        return ret

    # 生成规范LR(1)项集族 P167 图4-40
    def get_lr1_items(self) -> dict[ItemSet, None]:
        # 默认第一个产生式作为增广文法的起始状态，求闭包
        first_g: Generator = self._cfg.generators[0]
        first_i = Item(start=first_g.start, end=first_g.end, point=0, lf_symbol=self._cfg.end_mark)
        states = {self.get_closure(ItemSet([first_i])): None}
        new_states = {}

        # 如果是新的项集，返回True
        def merge_c(subItemSet):
            # 该状态（项集）已经存在
            if subItemSet in states:
                return False
            if subItemSet in new_states:
                return False
            # GOTO的结果是空的
            if len(subItemSet) == 0:
                return False
            new_states.setdefault(subItemSet, False)
            return True

        # 重复，直到没有新项集
        do_continue = True
        while do_continue:
            do_continue = False
            for itemSet in states:
                # 对于每个非终结符号
                for s in self._cfg.map_generators.keys():
                    do_continue |= merge_c(self.get_goto(itemSet, s))
                # 对于每个终结符号
                for s in self._cfg.end_symbols:
                    do_continue |= merge_c(self.get_goto(itemSet, s))
            for itemSet in new_states:
                states.setdefault(itemSet, None)

        return states

    # 求一个符号的First集 P140
    def get_first(self, X: Symbol) -> set[Symbol]:
        # 1. 如果X是一个终结符号或结束标记
        if X in self._cfg.end_symbols or X == self._cfg.end_mark:
            return {X}
        # 2. 如果X是一个非终结符号
        # 如果之前已经迭代过，返回
        if X in self._map_first.keys():
            return self._map_first[X]
        self._map_first[X] = set()
        # 缺少产生式
        if X not in self._cfg.map_generators.keys():
            raise ValueError(f'Not found generator of {X}')
        # 遍历X的产生式
        for g in self._cfg.map_generators[X]:
            # 从头开始，遇到终结符号或不带None的就结束
            for symbol in g.end:
                # 3. 如果产生式 X -> None
                if symbol.name == CONSTANT_GENERATOR_NONE:
                    self._map_first[X].add(symbol)
                    continue
                # 如果遇到终结符号
                if symbol in self._cfg.end_symbols:
                    self._map_first[X].add(symbol)
                    break
                subFirst = self.get_first(symbol)
                self._map_first[X] = self._map_first[X].union(subFirst)
                # 如果该非终结符号的first集不带None
                if Symbol(CONSTANT_GENERATOR_TYPE_END, CONSTANT_GENERATOR_NONE) not in subFirst:
                    break
        return self._map_first[X]

    # 求一个文法串的first集 P140
    def get_chain_first(self, chain: list[Symbol] or set[Symbol]) -> set[Symbol]:
        all_have_none = True
        symbol_none = Symbol(CONSTANT_GENERATOR_TYPE_END, CONSTANT_GENERATOR_NONE)
        ret = set()
        for s in chain:
            f_set = self.get_first(s)
            if symbol_none in f_set:
                f_set.remove(symbol_none)
                ret = ret.union(f_set)
            else:
                all_have_none = False
                ret = ret.union(f_set)
                break
        if all_have_none:
            ret.add(symbol_none)
        return ret

    # 求FOLLOW集 P140
    def get_follow(self, X: Symbol):
        if X not in self._map_follow.keys():
            self._get_all_follow()
        return self._map_follow[X]

    # 对符号A求一次follow集
    def _get_follow_once(self, A: Symbol) -> bool:
        sth_happened = False

        def merge_into_global(B: Symbol, subset: set[Symbol]):
            if B not in self._map_follow.keys():
                self._map_follow[B] = subset
                return True
            old_sz = len(self._map_follow[B])
            self._map_follow[B] = self._map_follow[B].union(subset)
            new_sz = len(self._map_follow[B])
            return new_sz > old_sz

        # 0. A必须是非终结符号
        if A in self._cfg.end_symbols:
            return False

        # 1. 如果X是开始符号，加入结束标记
        if A == self._cfg.start_mark:
            sth_happened |= merge_into_global(A, {self._cfg.end_mark})

        # 2. 如果A -> αBβ，把first(β)加入follow(b)
        # 遍历A的产生式
        for g in self._cfg.map_generators[A]:
            # \beta 的first集
            b_f = set()
            # 遍历产生式中不是最后一个的非终结符号
            for i in range(0, len(g.end) - 1):
                B = g.end[i]
                n_f = self.get_first(g.end[i + 1])
                # 如果后一个符号的first集含有none， 保存之前的
                if Symbol(CONSTANT_GENERATOR_TYPE_END, CONSTANT_GENERATOR_NONE) in n_f:
                    b_f = b_f.union(n_f)
                # 否则替换掉
                else:
                    b_f = n_f
                # 将文法串\beta 的first集加入follow(B)
                sth_happened |= merge_into_global(B, b_f)
        # 3. 如果A -> αB 或 A -> αBβ 且 first(β) 含ε，
        # 则follow(A)都在follow(B)中
        for g in self._cfg.map_generators[A]:
            for i in range(len(g.end) - 1, -1, -1):
                B = g.end[i]
                sth_happened |= merge_into_global(B, self._map_follow.get(A, set()))
                if Symbol(CONSTANT_GENERATOR_TYPE_END, CONSTANT_GENERATOR_NONE) not in self.get_first(B):
                    break

        return sth_happened

    # 计算所有非终结符号的follow集
    def _get_all_follow(self) -> None:
        non_final_sys = [key for key in self._cfg.map_generators.keys()]
        # 不断应用规则，直到再没有新的终结符号可以被加入到任意follow集合中为止
        do_continue = True
        while do_continue:
            do_continue = False
            for s in non_final_sys:
                do_continue |= self._get_follow_once(s)

    # 构造一个LR1语法分析表 P161
    def get_lr(self, deal_with_dangling_else=True):
        # 1. 构造规范LR(1)项集族
        c: list[ItemSet] = [key for key in self.get_lr1_items().keys()]
        ss = LRSheet(self._cfg, c)

        # 2. 构造语法分析动作
        # 2.1 ACTION[i, a] 为 "移入j"
        for state_i_idx in range(len(c)):
            state_i = c[state_i_idx]
            for item in state_i:
                # 后面没有符号
                if item.point >= len(item.end):
                    continue
                # a不是一个终结符号
                a = item.end[item.point]
                if a not in self._cfg.end_symbols:
                    continue
                state_j = self.get_goto(state_i, a)
                state_j_idx = c.index(state_j)
                if self._debug_mode:
                    print(f'2.1: from {state_i_idx} via {a} to {state_j_idx}')
                ss.set_action(state_i_idx, a, LRSheetAction(CONSTANT_ACTION_FORWARD_J, state_j_idx))
        # 2.2 ACTION[i, a] 为 "规约A -> \alpha" (A 不等于 增广文法的开始符号)
        for state_i_idx in range(len(c)):
            state_i = c[state_i_idx]
            for item in state_i:
                if item.point < len(item.end):
                    continue
                if item.start == self._cfg.start_mark:
                    continue
                # 特判：处理悬空else
                if deal_with_dangling_else:
                    # 后部开头是if && 马上要规约 && 后部不含else && 下一个字符是else
                    symbol_if = Symbol(CONSTANT_GENERATOR_TYPE_END, CONSTANT_TOKEN_NAME_KEYWORD_IF)
                    symbol_else = Symbol(CONSTANT_GENERATOR_TYPE_END, CONSTANT_TOKEN_NAME_KEYWORD_ELSE)
                    # if self._debug_mode:
                    #     print(f'd else: {item}')
                    if 1 < len(item.end) <= item.point \
                            and item.end[0] == symbol_if \
                            and symbol_else not in item.end \
                            and item.lf_symbol == symbol_else:
                        #  if self._debug_mode:
                        #      print(f' passed ')
                        continue
                symbol = item.lf_symbol
                if self._debug_mode:
                    print(f'2.2: from {state_i_idx} via {symbol} to {item.start}')
                ss.set_action(state_i_idx, symbol,
                              LRSheetAction(CONSTANT_ACTION_REDUCTION_A, [item.start, item]))
        # 2.3 ACTION[i, $]设置为接受
        for state_i_idx in range(len(c)):
            state_i = c[state_i_idx]
            for item in state_i:
                if item.point < len(item.end):
                    continue
                if item.start != self._cfg.start_mark:
                    continue
                ss.set_action(state_i_idx, self._cfg.end_mark, LRSheetAction(CONSTANT_ACTION_ACCEPT))
        # 3. GOTO[i, A] = j，A为非终结符号
        for state_i_idx in range(len(c)):
            state_i = c[state_i_idx]
            for A in self._cfg.map_generators.keys():
                state_j = self.get_goto(state_i, A)
                if state_j not in c:
                    continue
                state_j_idx = c.index(state_j)
                ss.set_goto(state_i_idx, A, state_j_idx)
        # 4. 其他默认报错
        # 5. 初始状态
        ss.set_initial_state(0)
        return ss

    # 用输入初始化格局
    def init_config(self, input_symbols):
        lr_sheet = self.get_lr()
        self._config = Configuration(parse_stk=[lr_sheet.state_idx],
                                     cur_symbols=[self._cfg.end_mark],
                                     input_symbols=input_symbols,
                                     lr_sheet=lr_sheet)

    @staticmethod
    def _deep_copy_symbol(obj: Symbol):
        newObj = Symbol(sType=obj.sType, content=obj.content, name=obj.name)
        od = obj.__dict__
        for key in od.keys():
            newObj.__setattr__(key, od[key])
        return newObj

    # 进行一步动作
    def forward(self, show=False):
        if self._config is None:
            raise ValueError('Configuration not initialized!')
        assert type(self._config) == Configuration
        action, l = self._config.forward(show=show)
        result = action.action_type == CONSTANT_ACTION_ACCEPT
        if l is not None:
            item, symbols = l
            if item.func is not None:
                s = self._config.cur_symbols[-1]
                s = Parser._deep_copy_symbol(s)
                item.func(symbols, s, self._sTable)
                self._config.cur_symbols[-1] = s
        self._config.show_current(action)
        return result

    # 进行所有剩下的语法分析
    def parse_all(self, show=False):
        do_continue = True
        while do_continue:
            do_continue = not self.forward(show)
        if show:
            print(self._sTable)
            print('--- 三地址码 ---')
            last_s = self._config.cur_symbols[1]
            last_code = last_s.__getattribute__(CONSTANT_SDD_ATTR_CODE)
            print('\n'.join(last_code))
