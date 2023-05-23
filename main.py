import argparse

from src import constants, grammar, parser, tokenizer, tester

# 创建CFG文法
def create_cfg():
    CONSTANT_GENERATOR_START_MARK = 'P\''
    CONSTANT_GENERATOR_END_MARK = '$'
    # 终结符号
    CONSTANT_GENERATOR_END_SYMBOLS = {constants.CONSTANT_TOKEN_NAME_IDENTIFIER,
                                      constants.CONSTANT_TOKEN_NAME_CONSTANT,
                                      constants.CONSTANT_GENERATOR_NONE} \
        .union(constants.CONSTANT_SET_KEYWORDS) \
        .union(constants.CONSTANT_SET_OPERATORS) \
        .union(constants.CONSTANT_SET_SEPARATORS)

    # 产生一条三地址代码
    def gen_instr(ls: list[str]):
        retStr = f'{" ".join(ls)}'
        return retStr

    tmp_name_cnt = 0
    debug_mode = True

    # 为三地址代码产生一个临时变量名
    def gen_tmp_name():
        nonlocal tmp_name_cnt
        tmp_name_cnt += 1
        return f'p{tmp_name_cnt}'

    # 为三地址代码产生一个临时标签
    tmp_label_cnt = 0
    def gen_tmp_label():
        nonlocal tmp_label_cnt
        tmp_label_cnt += 1
        return f'l{tmp_label_cnt}'

    # 返回递归子程序的闭包，执行的同时打印调试信息，未使用
    def sdd_wrapper(func):
        def sdd_func(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
            func(input_symbols, reduction_symbol, st)
            if debug_mode:
                print(f'sdd: {[str(s) for s in input_symbols]} -> {reduction_symbol} ')
        return sdd_func

    # 实现SDD语义的递归子程序
    def sdd_l(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_TYPE, input_symbols[0].content)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_WIDTH, 4)

    def sdd_d(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        st_type = input_symbols[0].__getattribute__(constants.CONSTANT_SDD_ATTR_TYPE)
        id_content = input_symbols[1].content
        st.put_entry(id_content, st_type)

    def sdd_s(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        s_code = []
        for i in range(len(input_symbols)):
            s = input_symbols[i]
            if s.content in CONSTANT_GENERATOR_END_SYMBOLS:
                continue
            s_code += s.__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, s_code)

    def sdd_p(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        p_code = []
        p_code += input_symbols[1].__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, p_code)

    def sdd_s0(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        s_id = input_symbols[0]
        result = input_symbols[2]

        e_code = result.__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        e_name = result.__getattribute__(constants.CONSTANT_SDD_ATTR_NAME)
        s_code = e_code + [gen_instr([s_id.content, '=', e_name])]
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, s_code)

    def sdd_t_lex(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        name = gen_tmp_name()
        val = input_symbols[0].content
        code = [gen_instr([name, '=', val])]

        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, code)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_NAME, name)

    def sdd_t_id(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        code = []

        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, code)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_NAME, input_symbols[0].content)

    def sdd_e_op(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        code = ['']
        name1 = input_symbols[0].__getattribute__(constants.CONSTANT_SDD_ATTR_NAME)
        name2 = input_symbols[2].__getattribute__(constants.CONSTANT_SDD_ATTR_NAME)
        name = gen_tmp_name()
        code[0] = gen_instr([name, '=', name1, input_symbols[1].content, name2])

        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_NAME, name)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, code)

    def sdd_e(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        code = input_symbols[0].__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        name = input_symbols[0].__getattribute__(constants.CONSTANT_SDD_ATTR_NAME)

        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, code)
        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_NAME, name)

    def sdd_if(input_symbols: list[grammar.Symbol], reduction_symbol: grammar.Symbol, st: parser.SymbolTable):
        s_next = gen_tmp_label()
        true_next = gen_tmp_label()
        false_next = gen_tmp_label()

        b = input_symbols[2]
        s1 = input_symbols[4]
        s2 = input_symbols[7]
        s1_next = s_next
        s2_next = s_next

        b_name = b.__getattribute__(constants.CONSTANT_SDD_ATTR_NAME)
        s_code = []
        s_code += b.__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        s_code += [f'if False {b_name} goto {false_next}']
        s_code += [f'{true_next}: '] + s1.__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        s_code += [gen_instr(['goto', s_next])]
        s_code += [f'{false_next}: '] + s2.__getattribute__(constants.CONSTANT_SDD_ATTR_CODE)
        s_code += [f'{s_next}:']

        reduction_symbol.__setattr__(constants.CONSTANT_SDD_ATTR_CODE, s_code)
        pass

    # 定义产生式
    c_generators = [
        grammar.ComposedGenerator('P\'', [['P']]), # 增广文法的第一个产生式
        grammar.ComposedGenerator('P', [['D', 'S']], func=sdd_p),
        grammar.ComposedGenerator('D', [[constants.CONSTANT_GENERATOR_NONE],
                                        ['D0', ';'],
                                        ['D0', ';', 'D']]),
        grammar.ComposedGenerator('D0', [['L', constants.CONSTANT_TOKEN_NAME_IDENTIFIER]], func=sdd_d),
        grammar.ComposedGenerator('L', [['int'], ['float']], func=sdd_l),
        grammar.ComposedGenerator('S', [['S0', ';'], ['S0', ';', 'S']], func=sdd_s),
        grammar.ComposedGenerator('S0', [[constants.CONSTANT_TOKEN_NAME_IDENTIFIER, '=', 'E']], func=sdd_s0),
        grammar.ComposedGenerator('S0', [['if', '(', 'C', ')', 'S0', ';', 'else', 'S0']], func=sdd_if),
        grammar.ComposedGenerator('T', [[constants.CONSTANT_TOKEN_NAME_CONSTANT]], func=sdd_t_lex),
        grammar.ComposedGenerator('T', [[constants.CONSTANT_TOKEN_NAME_IDENTIFIER]], func=sdd_t_id),
        grammar.ComposedGenerator('C', [['E', '>', 'E'], ['E', '<', 'E'], ['E', '==', 'E']], func=sdd_e_op),
        grammar.ComposedGenerator('E', [['T']], func=sdd_e),
        grammar.ComposedGenerator('E', [['E', '+', 'T'], ['E', '-', 'T']], func=sdd_e_op),
    ]

    return grammar.GrammarCFG(
        start_mark=CONSTANT_GENERATOR_START_MARK,
        end_mark=CONSTANT_GENERATOR_END_MARK,
        end_symbols=CONSTANT_GENERATOR_END_SYMBOLS,
        c_generators=c_generators
    )


# 运行单元测试
def test(te: tester.Tester):
    pass


if __name__ == '__main__':
    # 解析输入参数
    argParser = argparse.ArgumentParser(description='语法分析器 (SDD)')
    argParser.add_argument('--in', dest='input_file', type=str, help='输入程序文件')
    args = argParser.parse_args()
    # 获取CFG文法
    cfg = create_cfg()
    # 创建 Tokenizer 和 Parser
    parser = parser.Parser(cfg, debug_mode=False)
    tr = tokenizer.Tokenizer(input_path=args.input_file, cfg=cfg, debug_mode=False)
    # 运行测试
    # te = tester.Tester(cfg=cfg, parser=parser)
    # test(te)
    # 用所有输入初始化格局
    tokens = tr.read_all_left_tokens()
    parser.init_config(tokens)
    # 解析
    parser.parse_all(show=True)
