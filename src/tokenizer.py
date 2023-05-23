from .grammar import *


class Tokenizer:

    def __init__(self, input_path: str, cfg: GrammarCFG, debug_mode=False):
        self._buf_chars = []
        self._buf_tokens = []
        self._cfg = cfg
        self._f = open(input_path, 'r', encoding=FILE_ENCODING)
        self._debug_mode = debug_mode
        self.__name__ = 'Tokenizer'
        pass

    def _readOneChar(self):
        # if buf has chars, return one char
        if len(self._buf_chars) > 0:
            return self._buf_chars.pop(0)
        # else, read one char from file
        ch = self._f.read(1)
        if self._debug_mode:
            print(f'{self.__name__}: read char {ch}')
            pass
        if ch == '':
            raise EOFError
        return ch

    def hasMoreToken(self):
        # if buf has tokens, yes
        if len(self._buf_tokens) > 0:
            return True
        # else, try to read one token
        try:
            token = self._nextToken()
            self._buf_tokens.append(token)
        except EOFError:
            return False
        except Exception as e:
            print(f'{self.__name__}: error {e}')
            return False
        return True

    '''
    Return token's type & token itself.
    '''

    def _nextToken(self) -> [int, str]:
        # if buf has tokens, return one
        if len(self._buf_tokens):
            return self._buf_tokens.pop(0)
        ch = self._readOneChar()
        # ignore space
        while ch.isspace():
            ch = self._readOneChar()
        # ignore comment
        while ch == '#':
            while ch != '\n':
                ch = self._readOneChar()
            ch = self._readOneChar()
        # ignore space
        while ch.isspace():
            ch = self._readOneChar()
        # read one word
        tmp_token = ''
        # token is constant
        if ch.isdigit():
            while ch.isdigit():
                tmp_token += ch
                ch = self._readOneChar()
            self._buf_chars.append(ch)
            if False not in [ch.isdigit() for ch in tmp_token]:
                return [CONSTANT_TOKEN_TYPE_CONSTANT, tmp_token]
            return [CONSTANT_TOKEN_TYPE_UNKNOWN, tmp_token]
        # token is identifier or keyword
        if ch.isalpha() or ch == '_':
            while ch.isalpha() or ch == '_':
                tmp_token += ch
                ch = self._readOneChar()
            self._buf_chars.append(ch)
            # token is keyword
            if tmp_token in CONSTANT_SET_KEYWORDS:
                return [CONSTANT_TOKEN_TYPE_KEYWORD, tmp_token]
            # token is identifier
            if False not in [ch.isalpha() for ch in tmp_token]:
                return [CONSTANT_TOKEN_TYPE_IDENTIFIER, tmp_token]
            return [CONSTANT_TOKEN_TYPE_UNKNOWN, tmp_token]
        # token is other type
        while not ch.isspace():
            tmp_token += ch
            # token is separator
            if tmp_token in CONSTANT_SET_SEPARATORS:
                return [CONSTANT_TOKEN_TYPE_SEPARATOR, tmp_token]
            # token is operator
            if tmp_token in CONSTANT_SET_OPERATORS:
                ch = self._readOneChar()
                # fix fail to recognize >=
                if tmp_token + ch in CONSTANT_SET_OPERATORS:
                    tmp_token = tmp_token + ch
                else:
                    self._buf_chars.append(ch)
                return [CONSTANT_TOKEN_TYPE_OPERATOR, tmp_token]
            # continue
            ch = self._readOneChar()
        return [CONSTANT_TOKEN_TYPE_UNKNOWN, tmp_token]

    def nextToken(self) -> Symbol or None:
        try:
            token = self._nextToken()
            while token[0] == CONSTANT_TOKEN_TYPE_UNKNOWN:
                if self._debug_mode:
                    print(f'{self.__name__}: discard token {token}')
                token = self._nextToken()
            if self._debug_mode:
                print(f'{self.__name__}: read token {token}')
            # token 类型
            token_type = token[0]
            # token 内容
            token_content = token[1]
            # token 在产生式中的表达（决定是终结符号还是非终结符号）
            token_named_type = token_content
            if token_type == CONSTANT_TOKEN_TYPE_IDENTIFIER:
                token_named_type = CONSTANT_TOKEN_NAME_IDENTIFIER
            elif token_type == CONSTANT_TOKEN_TYPE_CONSTANT:
                token_named_type = CONSTANT_TOKEN_NAME_CONSTANT
            ret_symbol = Symbol(sType=CONSTANT_GENERATOR_TYPE_END,
                                name=token_named_type,
                                content=token_content)
            # 遗留问题：__eq__ 不会考虑token_content
            if ret_symbol not in self._cfg.end_symbols:
                ret_symbol.sType = CONSTANT_GENERATOR_TYPE_NONE_END
            return ret_symbol
        except EOFError:
            print(f'{self.__name__}: end of file')
            raise EOFError
        except Exception as e:
            raise e

    def read_all_left_tokens(self) -> list[Symbol]:
        tokens = []
        while self.hasMoreToken():
            tmp_token = self.nextToken()
            tokens.append(tmp_token)
        tokens.append(self._cfg.end_mark)
        return tokens
