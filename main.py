import argparse

from src import constants, grammar, parser, tokenizer, tester


def create_cfg():
    CONSTANT_GENERATOR_START_MARK = 'P\''
    CONSTANT_GENERATOR_END_MARK = '$'
    CONSTANT_GENERATOR_END_SYMBOLS = {constants.CONSTANT_TOKEN_NAME_IDENTIFIER,
                                      constants.CONSTANT_TOKEN_NAME_CONSTANT,
                                      constants.CONSTANT_GENERATOR_NONE} \
        .union(constants.CONSTANT_SET_KEYWORDS) \
        .union(constants.CONSTANT_SET_OPERATORS) \
        .union(constants.CONSTANT_SET_SEPARATORS)

 
    c_generators = [
        grammar.ComposedGenerator('P\'', [['P']]),
        grammar.ComposedGenerator('P', [['D', 'S']]),
        grammar.ComposedGenerator('D', [[constants.CONSTANT_GENERATOR_NONE],
                                        ['D0', ';'],
                                        ['D0', ';', 'D']]),
        grammar.ComposedGenerator('D0', [['L', constants.CONSTANT_TOKEN_NAME_IDENTIFIER]]),
        grammar.ComposedGenerator('L', [['int'], ['float']]),
        grammar.ComposedGenerator('S', [['S0', ';'], ['S0', ';', 'S']]),
        grammar.ComposedGenerator('S0', [[constants.CONSTANT_TOKEN_NAME_IDENTIFIER, '=', 'E']]),
        grammar.ComposedGenerator('S0', [['if', '(', 'C', ')', 'S0', ';', 'else', 'S0']]),
        grammar.ComposedGenerator('T', [[constants.CONSTANT_TOKEN_NAME_CONSTANT]]),
        grammar.ComposedGenerator('T', [[constants.CONSTANT_TOKEN_NAME_IDENTIFIER]]),
        grammar.ComposedGenerator('C', [['E', '>', 'E'], ['E', '<', 'E'], ['E', '==', 'E']]),
        grammar.ComposedGenerator('E', [['T']]),
        grammar.ComposedGenerator('E', [['E', '+', 'T'], ['E', '-', 'T']]),
    ]

    return grammar.GrammarCFG(
        start_mark=CONSTANT_GENERATOR_START_MARK,
        end_mark=CONSTANT_GENERATOR_END_MARK,
        end_symbols=CONSTANT_GENERATOR_END_SYMBOLS,
        c_generators=c_generators
    )


def test(te: tester.Tester):
    pass


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description='语法分析器')
    argParser.add_argument('--in', dest='input_file', type=str, help='输入程序文件')
    args = argParser.parse_args()

    cfg = create_cfg()
    parser = parser.Parser(cfg, debug_mode=False)
    tr = tokenizer.Tokenizer(input_path=args.input_file, cfg=cfg, debug_mode=False)

    # te = tester.Tester(cfg=cfg, parser=parser)
    # test(te)

    tokens = tr.read_all_left_tokens()
    parser.init_config(tokens)
    parser.parse_all(show=True)
