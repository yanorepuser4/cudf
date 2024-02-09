/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/types.hpp>
#include <thrust/optional.h>
#include <rmm/device_vector.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * Json token enum
 */
enum class json_token {
  // start token
  INIT = 0,

  // successfully parsed the whole Json string
  SUCCESS,

  // get error when parsing Json string
  ERROR,

  // '{'
  START_OBJECT,

  // '}'
  END_OBJECT,

  // '['
  START_ARRAY,

  // ']'
  END_ARRAY,

  // e.g.: key1 in {"key1" : "value1"}
  FIELD_NAME,

  // e.g.: value1 in {"key1" : "value1"}
  VALUE_STRING,

  // e.g.: 123 in {"key1" : 123}
  VALUE_NUMBER_INT,

  // e.g.: 1.25 in {"key1" : 1.25}
  VALUE_NUMBER_FLOAT,

  // e.g.: true in {"key1" : true}
  VALUE_TRUE,

  // e.g.: false in {"key1" : false}
  VALUE_FALSE,

  // e.g.: null in {"key1" : null}
  VALUE_NULL

};

/**
 * options for json_parser
 *
 */
class json_parser_options {
 public:
  CUDF_HOST_DEVICE inline bool get_allow_single_quotes() const { return allow_single_quotes; }

  CUDF_HOST_DEVICE inline bool get_allow_unescaped_control_chars() const
  {
    return allow_unescaped_control_chars;
  }

  CUDF_HOST_DEVICE inline bool get_max_nesting_depth() const { return max_nesting_depth; }

  void set_allow_single_quotes(bool _allow_single_quotes)
  {
    allow_single_quotes = _allow_single_quotes;
  }

  void set_allow_unescaped_control_chars(bool _allow_unescaped_control_chars)
  {
    allow_unescaped_control_chars = _allow_unescaped_control_chars;
  }

 private:
  // if true, allow both ' and " for quoting strings. e.g.: json string {'k1' : "v1"} is valid
  // if true, Json is not valid formal format
  bool allow_single_quotes = false;

  // Feature that determines whether parser will allow JSON Strings to contain unescaped control
  // characters (ASCII characters with value less than 32, including tab and line feed characters)
  // or not.
  // e.g., how to represent carriage return and newline character characters:
  //   if false, "\\n\\r" is allowed. There are 4 chars(\ n \ r), after escaped, 2 chars remains.
  //     Note: print "\\n\\r" => \n\r
  //   if true, allow "\n\r" string
  //     Note: in this string only contains 2 charactors:
  //       carriage return character (ASCII code 13)
  //       newline character (ASCII code 10)
  bool allow_unescaped_control_chars = false;

  // it's from Jackson json
  int max_nesting_depth = 1000;

  // it's from Jackson json
  // currently not take effective
  int max_num_len = 1000;

  // it's from Jackson json
  // currently not take effective
  int max_string_len = 20000000;
};

/**
 * Json parser, provides token by token parsing.
 * Follow Jackson Json format by default.
 *
 *
 * For Json format:
 * Refer to https://www.json.org/json-en.html.
 *
 * Note: when setting `allow_single_quotes` `allow_unescaped_control_chars`, then
 * format is invalid formal Json format.
 *
 * White space can only be 4 chars: ' ', '\n', '\r', '\t',
 * Jackson does not allow other control chars as white spaces.
 *
 * Valid number examples:
 *   0, 102, -0, -102, 0.3, -0.3
 *   1e-5, 1E+5, 1e0, 1E0, 1.3e5
 *   1e01    allow leading zeor after 'e'
 *
 * Invalid number examples:
 *   00, -00   Leading zeroes not allowed
 *   infinity, +infinity, -infinity
 *   1e, 1e+, 1e-, -1., 1.
 *
 * When allow_single_quotes is true:
 *   Valid string examples:
 *     "\'" , "\"" ,  '\'' , '\"' , '"' , "'"
 *
 *  When allow_single_quotes is false:
 *   Inalid string examples:
 *     "\'"
 *
 *  When allow_unescaped_control_chars is true:
 *    Valid string: "asscii_control_chars"
 *      here `asscii_control_chars` represents control chars which in Ascii code range: [0, 32)
 *
 *  When allow_unescaped_control_chars is false:
 *    Invalid string: "asscii_control_chars"
 *      here `asscii_control_chars` represents control chars which in Ascii code range: [0, 32)
 *
 */

class json_parser {
 public:
  CUDF_HOST_DEVICE inline json_parser(json_parser_options const _options,
                                      char const* const _json_start_pos,
                                      int64_t const _json_len)
    : options(_options),
      json_start_pos(_json_start_pos),
      curr_pos(_json_start_pos),
      json_end_pos(_json_start_pos + _json_len)
  {
  }

 private:
  /**
   * is current position EOF?
   */
  CUDF_HOST_DEVICE inline bool eof() { return curr_pos >= json_end_pos; }

  /**
   * is hex digits: 0-9, A-F, a-f
   */
  CUDF_HOST_DEVICE inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  /**
   * is 0 to 9 digit
   */
  CUDF_HOST_DEVICE inline bool is_digit(char c) { return (c >= '0' && c <= '9'); }

  /**
   * is white spaces: ' ', '\t', '\n' '\r'
   */
  CUDF_HOST_DEVICE inline bool is_whitespace(char c)
  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  /**
   * skips 4 characters: ' ', '\t', '\n' '\r'
   */
  CUDF_HOST_DEVICE inline void skip_whitespaces()
  {
    while (!eof() && is_whitespace(*curr_pos)) {
      curr_pos++;
    }
  }

  /**
   * check current char, if it's expected, then plus the position
   */
  CUDF_HOST_DEVICE inline bool try_skip(char expected)
  {
    if (!eof() && *curr_pos == expected) {
      curr_pos++;
      return true;
    }
    return false;
  }

  /**
   * get a bit value from a long
   */
  CUDF_HOST_DEVICE inline bool get_bit(int64_t number, int position)
  {
    int64_t shifted = number >> position;
    return shifted & 1;
  }

  /**
   * set a bit value from a long
   */
  CUDF_HOST_DEVICE inline void set_bit(int64_t& number, int position)
  {
    int64_t mask = 1L << position;

    // Perform bitwise OR to set the bit at the position
    number |= mask;
  }

  /**
   * reset a bit value from a long
   */
  CUDF_HOST_DEVICE inline void reset_bit(int64_t& number, int position)
  {
    int64_t mask = 1L << position;

    // Perform bitwise OR to set the bit at the position
    number &= mask;
  }

  /**
   * if depth exceeds limitation, return false;
   */
  CUDF_HOST_DEVICE inline bool try_push_context(json_token token)
  {
    if (context_stack_depth() < options.get_max_nesting_depth()) {
      push_context(token);
      return true;
    }

    return false;
  }

  /**
   * record the nested state into stack: Json object or Json array
   * if stack depth <= 64, save to a long value, each bit save a context;
   * if stack depth >  64, save to a stack.
   */
  CUDF_HOST_DEVICE inline void push_context(json_token token)
  {
    bool v = (json_token::START_OBJECT == token ? true : false);
    if (nested_context_stack.has_value()) {
      nested_context_stack->push_back(v);
    } else {
      if (stack_size_for_long < 64) {
        if (v) {
          set_bit(context_stack_long, stack_size_for_long++);
        } else {
          reset_bit(context_stack_long, stack_size_for_long++);
        }
      } else {
        // nested depth is 64, copy contexts into a stack
        nested_context_stack.emplace(rmm::device_vector<bool>(128));
        for (size_t i = 0; i < 64; i++) {
          nested_context_stack->push_back(get_bit(context_stack_long, i));
        }
        // add the token
        push_context(token);
      }
    }
  }

  /**
   * whether the top of nested context stack is Json object context
   * true is object, false is array
   * only has two contexts: object or array
   */
  CUDF_HOST_DEVICE inline bool is_object_context()
  {
    if (nested_context_stack.has_value()) {
      return nested_context_stack->back();
    } else {
      return get_bit(context_stack_long, stack_size_for_long - 1);
    }
  }

  /**
   * pop top context from stack
   */
  CUDF_HOST_DEVICE inline void pop_curr_context()
  {
    if (nested_context_stack.has_value()) {
      nested_context_stack->pop_back();
    } else {
      stack_size_for_long--;
    }
  }

  /**
   * get nested stack depth
   */
  CUDF_HOST_DEVICE inline int context_stack_depth()
  {
    if (nested_context_stack.has_value()) {
      return nested_context_stack->size();
    } else {
      return stack_size_for_long;
    }
  }

  /**
   * is context stack is empty
   */
  CUDF_HOST_DEVICE inline bool is_context_stack_empty() { return context_stack_depth() == 0; }

  /**
   * parse the first token in current value
   * e.g., after done this function:
   *   current token is START_OBJECT if current value is object
   *   current token is START_ARRAY if current value is array
   *   current token is string/num/true/false/null if current value is terminal
   *   current token is ERROR if parse failed
   */
  CUDF_HOST_DEVICE inline void parse_first_token_in_value()
  {
    if (!eof()) {
      // non eof
      char c = *curr_pos;
      switch (c) {
        case '{':
          if (!try_push_context(json_token::START_OBJECT)) {
            curr_token = json_token::ERROR;
            return;
          }
          curr_pos++;
          curr_token = json_token::START_OBJECT;
          break;

        case '[':
          if (!try_push_context(json_token::START_ARRAY)) {
            curr_token = json_token::ERROR;
            return;
          }
          curr_pos++;
          curr_token = json_token::START_ARRAY;
          break;

        case '"': parse_double_quoted_string(); break;

        case '\'':
          if (options.get_allow_single_quotes()) {
            parse_single_quoted_string();
          } else {
            curr_token = json_token::ERROR;
          }
          break;

        case 't':
          curr_pos++;
          parse_true();
          break;

        case 'f':
          curr_pos++;
          parse_false();
          break;

        case 'n':
          curr_pos++;
          parse_null();
          break;

        default: parse_number();
      }
    } else {
      // eof
      curr_token = json_token::ERROR;
    }
  }

  // =========== Parse string begin ===========

  /**
   * parse ' quoted string
   */
  CUDF_HOST_DEVICE inline void parse_single_quoted_string()
  {
    if (try_parse_single_quoted_string()) {
      curr_token = json_token::VALUE_STRING;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse ' quoted string
   */
  CUDF_HOST_DEVICE inline void parse_double_quoted_string()
  {
    if (try_parse_double_quoted_string()) {
      curr_token = json_token::VALUE_STRING;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /*
   * try parse ' or " quoted string
   * when allow single quote, first try single quote
   */
  CUDF_HOST_DEVICE inline bool try_parse_string()
  {
    if (options.get_allow_single_quotes() && *curr_pos == '\'') {
      return try_parse_single_quoted_string();
    } else {
      return try_parse_double_quoted_string();
    }
  }

  /**
   * try parse ' quoted string
   */
  CUDF_HOST_DEVICE inline bool try_parse_single_quoted_string() { return try_parse_string('\''); }

  /**
   * try parse " quoted string
   */
  CUDF_HOST_DEVICE inline bool try_parse_double_quoted_string() { return try_parse_string('\"'); }

  /**
   * try parse quoted string using passed `quote_char`
   * `quote_char` can be ' or "
   * For UTF-8 encoding:
   *   Single byte char: The most significant bit of the byte is always 0
   *   Two-byte characters: The leading bits of the first byte are 110,
   *     and the leading bits of the second byte are 10.
   *   Three-byte characters: The leading bits of the first byte are 1110,
   *     and the leading bits of the second and third bytes are 10.
   *   Four-byte characters: The leading bits of the first byte are 11110,
   *     and the leading bits of the second, third, and fourth bytes are 10.
   * Because Json structural chars([ ] { } , :), string quote char(" ') and
   * Escape char \ are all Ascii(The leading bit is 0), so it's safe that do
   * not convert byte array to UTF-8 char.
   *
   * When quote is " and allow_unescaped_control_chars is false, grammar is:
   *
   *   STRING
   *     : '"' (ESC | SAFECODEPOINT)* '"'
   *     ;
   *
   *   fragment ESC
   *     : '\\' (["\\/bfnrt] | UNICODE)
   *     ;
   *
   *   fragment UNICODE
   *     : 'u' HEX HEX HEX HEX
   *     ;
   *
   *   fragment HEX
   *     : [0-9a-fA-F]
   *     ;
   *
   *   fragment SAFECODEPOINT
   *       // 1 not " or ' depending to allow_single_quotes
   *       // 2 not \
   *       // 3 non control character: Ascii value not in [0, 32)
   *     : ~ ["\\\u0000-\u001F]
   *     ;
   *
   * When allow_unescaped_control_chars is true:
   *   Allow [0-32) control Ascii chars without escape
   * When allow_single_quotes is true:
   *   These strings are allowed: '\'' , '\"' , '"' , "\"" , "\'" , "'"
   */
  CUDF_HOST_DEVICE inline bool try_parse_string(char quote_char)
  {
    if (!try_skip(quote_char)) { return false; }

    // scan string content
    while (!eof()) {
      char c = *curr_pos;
      if (c == quote_char) {
        // path 1: close string
        curr_pos++;
        return true;
      } else if (c >= 0 && c < 32 && options.get_allow_unescaped_control_chars()) {
        // path 2: unescaped control char
        curr_pos++;
        continue;
      } else {
        switch (c) {
          case '\\':
            // path 3: escape path
            curr_pos++;
            if (!try_skip_escape_part()) { return false; }
            break;
          default:
            // path 4: safe code point
            if (!try_skip_safe_code_point(c)) { return false; }
        }
      }
    }

    return false;
  }

  /**
   * skip the second char in \", \', \\, \/, \b, \f, \n, \r, \t;
   * skip the HEX chars in \u HEX HEX HEX HEX.
   */
  CUDF_HOST_DEVICE inline bool try_skip_escape_part()
  {
    // already skiped the first \

    // try skip second part
    if (!eof()) {
      switch (*curr_pos) {
        case '\"': curr_pos++; return true;
        case '\'':
          // only allow escape ' when `allow_single_quotes`
          curr_pos++;
          return options.get_allow_single_quotes();
        case '\\':
        case '/':
        case 'b':
        case 'f':
        case 'n':
        case 'r':
        case 't':
          // path 1: \", \', \\, \/, \b, \f, \n, \r, \t
          curr_pos++;
          return true;
        case 'u':
          // path 2: \u HEX HEX HEX HEX
          curr_pos++;
          if (!try_skip_unicode()) {
            return false;
          } else {
            return false;
          }
        default:
          // path 3: invalid
          return false;
      }
    } else {
      // eof, no escaped char after char '\'
      return false;
    }
  }

  /**
   * parse:
   *   fragment SAFECODEPOINT
   *       // 1 not " or ' depending to allow_single_quotes
   *       // 2 not \
   *       // 3 non control character: Ascii value not in [0, 32)
   *     : ~ ["\\\u0000-\u001F]
   *     ;
   */
  CUDF_HOST_DEVICE inline bool try_skip_safe_code_point(char c)
  {
    // 1 the char is not quoted/close char, here satisfy, do not need to check again

    // 2. the char is not \, here satisfy, do not need to check again

    // 3. chars not in [0, 32)
    if (!(c >= 0 && c < 32)) {
      curr_pos++;
      return true;
    } else {
      return false;
    }
  }

  /**
   * try skip 4 HEX chars
   * in pattern: '\\' 'u' HEX HEX HEX HEX
   */
  CUDF_HOST_DEVICE inline bool try_skip_unicode()
  {
    // already parsed u

    if (try_skip_hex() && try_skip_hex() && try_skip_hex() && try_skip_hex()) { return true; }
    return false;
  }

  /**
   * try skip HEX
   */
  CUDF_HOST_DEVICE inline bool try_skip_hex()
  {
    if (!eof() && is_hex_digit(*curr_pos)) {
      curr_pos++;
      return true;
    }
    return false;
  }

  // =========== Parse string end ===========

  // =========== Parse number begin ===========

  /**
   * parse number, grammar is:
   * NUMBER
   *   : '-'? INT ('.' [0-9]+)? EXP?
   *   ;
   *
   * fragment INT
   *   // integer part forbis leading 0s (e.g. `01`)
   *   : '0'
   *   | [1-9] [0-9]*
   *   ;
   *
   * fragment EXP
   *   : [Ee] [+\-]? [0-9]+
   *   ;
   *
   * valid number:    0, 0.3, 0e005, 0E005
   * invalid number:  0., 0e, 0E
   *
   */
  CUDF_HOST_DEVICE inline void parse_number()
  {
    // parse sign
    try_skip('-');

    // parse unsigned number
    bool is_float = false;
    if (try_unsigned_number(is_float)) {
      curr_token = (is_float ? json_token::VALUE_NUMBER_FLOAT : json_token::VALUE_NUMBER_INT);
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse:  INT ('.' [0-9]+)? EXP?
   *
   * @param[out] is_float, if contains `.` or `e`, set true
   */
  CUDF_HOST_DEVICE inline bool try_unsigned_number(bool& is_float)
  {
    if (!eof()) {
      char c = *curr_pos;
      if (c >= '1' && c <= '9') {
        curr_pos++;
        // first digit is [1-9]
        // path: INT = [1-9] [0-9]*
        skip_zero_or_more_digits();
        return parse_number_from_fraction(is_float);
      } else if (c == '0') {
        curr_pos++;
        // first digit is [0]
        // path: INT = '0'
        return parse_number_from_fraction(is_float);
      } else {
        // first digit is non [0-9]
        return false;
      }
    } else {
      // eof, has no digits
      return false;
    }
  }

  /**
   * parse: ('.' [0-9]+)? EXP?
   */
  CUDF_HOST_DEVICE inline bool parse_number_from_fraction(bool& is_float)
  {
    // parse fraction
    if (try_skip('.')) {
      // has fraction
      is_float = true;
      // try pattern: [0-9]+
      if (!try_skip_one_or_more_digits()) { return false; }
    }

    // parse exp
    if (!eof() && (*curr_pos == 'e' || *curr_pos == 'E')) {
      curr_pos++;
      is_float = true;
      return try_parse_exp();
    }

    return true;
  }

  /**
   * parse: [0-9]*
   * skip zero or more [0-9]
   */
  CUDF_HOST_DEVICE inline void skip_zero_or_more_digits()
  {
    while (!eof()) {
      if (is_digit(*curr_pos)) {
        curr_pos++;
      } else {
        // point to first non-digit char
        break;
      }
    }
  }

  /**
   * parse: [0-9]+
   * try skip one or more [0-9]
   */
  CUDF_HOST_DEVICE inline bool try_skip_one_or_more_digits()
  {
    if (!eof() && is_digit(*curr_pos)) {
      curr_pos++;
      skip_zero_or_more_digits();
      return true;
    } else {
      return false;
    }
  }

  /**
   * parse [eE][+-]?[0-9]+
   */
  CUDF_HOST_DEVICE inline bool try_parse_exp()
  {
    // already parsed [eE]

    // parse [+-]?
    if (!eof() && (*curr_pos == '+' || *curr_pos == '-')) { curr_pos++; }

    // parse [0-9]+
    return try_skip_one_or_more_digits();
  }

  // =========== Parse number end ===========

  /**
   * parse true
   */
  CUDF_HOST_DEVICE inline void parse_true()
  {
    if (try_skip('r') && try_skip('u') && try_skip('e')) {
      curr_token = json_token::VALUE_TRUE;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse false
   */
  CUDF_HOST_DEVICE inline void parse_false()
  {
    if (try_skip('a') && try_skip('l') && try_skip('s') && try_skip('e')) {
      curr_token = json_token::VALUE_FALSE;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse null
   */
  CUDF_HOST_DEVICE inline void parse_null()
  {
    if (try_skip('u') && try_skip('l') && try_skip('l')) {
      curr_token = json_token::VALUE_NULL;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * parse the key string in key:value pair
   */
  CUDF_HOST_DEVICE inline void parse_field_name()
  {
    if (try_parse_string()) {
      curr_token = json_token::FIELD_NAME;
    } else {
      curr_token = json_token::ERROR;
    }
  }

  /**
   * continute parsing the next token and update curent token
   * Note: only parse one token at a time
   */
  CUDF_HOST_DEVICE inline json_token parse_next_token()
  {
    if (curr_token == json_token::SUCCESS || curr_token == json_token::ERROR) {
      CUDF_FAIL("Parsing is done, try parse again.", std::invalid_argument);
    }

    skip_whitespaces();

    if (!eof()) {
      char c = *curr_pos;
      if (is_context_stack_empty()) {
        // stack is empty

        if (curr_token == json_token::INIT) {
          // main root entry point
          parse_first_token_in_value();
        } else {
          // privious token is not INIT, means already get a token; stack is empty;
          // Json should be a terminal value without any nested object/array.
          // Here try to get more token, it's wrong.
          curr_token = json_token::ERROR;
        }
      } else {
        // stack is non-empty

        if (is_object_context()) {
          // in Json object context
          if (curr_token == json_token::START_OBJECT) {
            // previous token is '{'
            if (c == '}') {
              // empty object
              // close curr object context
              curr_pos++;
              curr_token = json_token::END_OBJECT;
              pop_curr_context();
            } else {
              // parse key in key:value pair
              parse_field_name();
            }
          } else if (curr_token == json_token::FIELD_NAME) {
            if (c == ':') {
              // skip ':' and parse value in key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value();
            } else {
              curr_token = json_token::ERROR;
            }
          } else {
            // expect next key:value pair or '}'
            if (c == '}') {
              // end of object
              curr_pos++;
              curr_token = json_token::END_OBJECT;
              pop_curr_context();
            } else if (c == ',') {
              // parse next key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_field_name();
            } else {
              curr_token = json_token::ERROR;
            }
          }
        } else {
          // in Json array context
          if (curr_token == json_token::START_ARRAY) {
            // previous token is '['
            if (c == ']') {
              // curr: ']', empty array
              curr_pos++;
              curr_token = json_token::END_ARRAY;
              pop_curr_context();
            } else {
              // non-empty array, parse the first value in the array
              parse_first_token_in_value();
            }
          } else {
            if (c == ',') {
              // skip ',' and parse the next value
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value();
            } else if (c == ']') {
              // end of array
              curr_pos++;
              curr_token = json_token::END_ARRAY;
              pop_curr_context();
            } else {
              curr_token = json_token::ERROR;
            }
          }
        }
      }
    } else {
      // eof
      if (is_context_stack_empty() && curr_token != json_token::INIT) {
        // eof and stack is empty
        curr_token = json_token::SUCCESS;
      } else {
        // eof but has unclosed Json array or Json object
        curr_token = json_token::ERROR;
      }
    }
    return curr_token;
  }

 public:
  /**
   * continute parsing, get next token.
   * The final tokens are ERROR or SUCCESS;
   */
  CUDF_HOST_DEVICE json_token next_token() { return parse_next_token(); }

  /**
   * is valid Json by parsing through all tokens
   */
  CUDF_HOST_DEVICE bool is_valid()
  {
    while (curr_token != json_token::ERROR && curr_token != json_token::SUCCESS) {
      next_token();
    }
    return curr_token == json_token::SUCCESS;
  }

 private:
  json_parser_options const options;
  char const* const json_start_pos{nullptr};
  char const* const json_end_pos{nullptr};

  char const* curr_pos{nullptr};
  json_token curr_token{json_token::INIT};

  // saves the nested contexts: Json object context or Json array context
  // true is Json object context; false is Json array context
  // When encounter EOF and this stack is non-empty, then parsing will fail
  // if depth <= 64, use a long record context
  // if depth is > 64, use stack to save
  int64_t context_stack_long                                         = 0;
  int32_t stack_size_for_long                                        = 0;
  thrust::optional<rmm::device_vector<bool>> nested_context_stack = thrust::nullopt;

  int64_t int_value;
  double double_value;
  std::string string_value;
};

}  // namespace detail
}  // namespace strings
}  // namespace cudf
