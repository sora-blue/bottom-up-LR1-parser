from .parser import *


class Tester:
    def __init__(self, cfg: GrammarCFG, parser: Parser):
        self.cfg = cfg
        self.parser = parser
        self.non_end_symbols = [s for s in cfg.map_generators.keys()]
        self.end_symbols = [s for s in cfg.end_symbols]
        self.all_symbols = [s for s in cfg.end_symbols] + [s for s in cfg.map_generators.keys()]

    def test_closure(self):
        cfg = self.cfg
        parser = self.parser
        for idx in range(1):
            g = cfg.generators[idx]
            i = ItemSet([Item(g.start, g.end, 0, cfg.end_mark)])
            c = parser.get_closure(i)
            print(f'using {i.toList()[0]}')
            print(c)

    def test_lr1_items(self):
        d = self.parser.get_lr1_items()
        states = [key for key in d.keys()]
        for i in range(len(states)):
            print(f'using state {i}: ')
            print(states[i])

    def test_lr1_state_machine(self):
        d = self.parser.get_lr1_items()
        states = [key for key in d.keys()]
        # states
        print(f'---states---\n')
        self.test_lr1_items()
        # links
        print(f'---links---\n')
        for i in range(len(states)):
            for s in self.all_symbols:
                n_i = self.parser.get_goto(states[i], s)
                # Goto 值为空
                if len(n_i) == 0:
                    continue
                for j in range(len(states)):
                    if n_i == states[j]:
                        print(f'using from state {i} to state {j} via {s}')

    def test_first(self):
        def print_first(symbols):
            for s in symbols:
                print(f'using first({s}): ')
                r = self.parser.get_first(s)
                print(f'{" ".join([str(i) for i in r])}\n', )

        print_first(self.non_end_symbols)
        print(SPLIT_LINE)
        print_first(self.end_symbols)

    def test_follow(self):
        for s in self.non_end_symbols:
            r = self.parser.get_follow(s)
            print(f'using follow({s}): ')
            print(f'{" ".join([str(i) for i in r])}\n')

    def test_slr_sheet(self):
        self.parser.get_lr()





