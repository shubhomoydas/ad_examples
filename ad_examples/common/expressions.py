import tokenize
import re
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

"""
General Rule-parsing functions. We might use only a subset of the features available.

For some rule illustrations/examples, see test_rule_apis()

To test:
    pythonw -m ad_examples.common.expressions
"""

# Supported datatypes
DTYPE_CATEGORICAL = 0 # categorical
DTYPE_CONTINUOUS  = 1 # numerical float values
DTYPE_BINARY  = 2 # categorical {'0','1'}

LABEL_VAR_INDEX = -1
UNINITIALIZED_VAR_INDEX = -2
ILLEGAL_VAR_INDEX = -3
UNKNOWN_CATEGORICAL_VALUE_INDEX = -1

DEFAULT_PREDICATE_WEIGHT = 1.0


class stack(list):
    def push(self, item):
        self.append(item)

    def is_empty(self):
        return not self


class DType(object):

    def __init__(self):
        self.name = None

    def is_numeric(self):
        pass

    def is_continuous(self):
        pass


class Factor(DType):
    """Stores information about the values that a categorical variable can take.

    Also provides the one-hot encodings

    Attributes:
        __values: dict
            Mapping of categorical values from string representations
            to integer.
        __fromindex: dict
            Mapping of integers to corresponding categorical string
            representations.
        __onehot: dict
            Mapping of categorical (integer) values to corresponding
            one-hot representations.
        __onehotNA: np.array
            One-hot vector that will be returned when the value is
            missing. This has 'nan' in all positions of the vector.

    """

    def __init__(self, vals, sort=True):
        """Creates a Factor instance from the input set of values.

        Args:
            vals: list
                An unique set of allowable values/levels for the factor.
            sort: boolean, (default True)
                Whether to sort the values alphabetically before assigning
                them indexes. Sometimes the input order might need to be
                maintained (with sort=False) e.g., if these represent
                column names which occur in the specific input order.
        """

        DType.__init__(self)

        self.__values = {}
        self.__fromindex = {}
        self.__onehot = {}
        tmp = [x for x in vals if x != '']
        if sort:
            tmp = sorted(tmp)
        self.__onehotNA = np.empty(len(tmp))
        self.__onehotNA.fill(np.nan)
        tmphot = np.zeros(len(tmp), dtype=float)
        for i in range(0, len(tmp)):
            self.__values[tmp[i]] = i
            self.__fromindex[i] = tmp[i]
            self.__onehot[i] = np.array(tmphot)  # store a new copy
            self.__onehot[i][i] = 1

    def is_numeric(self):
        return False

    def is_continuous(self):
        return False

    def all_values(self):
        return self.__values

    def index_of(self, value):
        return self.__values.get(value)

    def encode_to_one_hot(self, index):
        """Encode the categorical variable to one-hot vector.

        Some algorithms like scikit-learn decision trees need the data
        to be presented only as numeric vectors. In that case, we need
        to encode all categorical features as one-hot vectors.

        Args:
            index: int
                Index of the value

        Returns: np.array
        """

        ret = self.__onehot.get(index)
        return self.__onehotNA if ret is None else ret

    def __getitem__(self, index):
        return self.__fromindex[index]

    def num_values(self):
        return len(self.__values)

    def __repr__(self):
        return 'Factor ' + repr(self.__values)

    def __str__(self):
        return 'Factor ' + repr(self.__values)


class NumericContinuous(DType):
    """Definitions for Gaussian distributed real values."""

    def __init__(self, vals=None):
        """Initializes the mean and variance of the Gaussian variable."""

        DType.__init__(self)

        if vals is None:
            vals = [0, 1]  # some dummy. This is more for information.

        # Ignore NaNs
        n = np.count_nonzero(~np.isnan(vals))
        if n > 0:
            self.mean = np.nanmean(vals)
            self.variance = np.nanvar(vals)
        else:
            self.mean = 0
            self.variance = 0

    def is_numeric(self):
        return True

    def is_continuous(self):
        return True

    def __repr__(self):
        return 'Continuous(mean=' + str(self.mean) + ", var=" + str(self.variance) + ")"

    def __str__(self):
        return 'Continuous(mean=' + str(self.mean) + ", var=" + str(self.variance) + ")"


class FeatureMetadata(object):
    """Contains all metadata related to features.

    Attributes:
        lblname: string
            The column name of the label column.
        lbldef: Factor
            All permissible label values.
        featurenames: Factor
            All feature names stored in the same order as features.
        featuredefs: list
            Contains info about each feature.

    """

    def __init__(self, lblname=None, lbldef=None,
                 featurenames=None, featuredefs=None):
        self.lblname = lblname
        self.lbldef = lbldef
        self.featurenames = featurenames
        self.featuredefs = featuredefs

    def num_features(self):
        return 0 if self.featuredefs is None else len(self.featuredefs)

    def _tostr(self):
        return "[FeatureMetadata]\nlblname: " + str(self.lblname) + "\n" + \
               "lbldef: " + str(self.lbldef) + "\n" + \
               "featurenames: " + str(self.featurenames) + "\n" + \
               "featuredefs: " + str(self.featuredefs)

    def __repr__(self):
        return self._tostr()

    def __str__(self):
        return self._tostr()


class Expression(object):
    """Base class for any expression that needs to be parsed or evaluated."""

    def evaluate(self, inst, lbl, meta):
        """Given an instance, will evaluate the expression.

        The expression might return a literal or a True/False value.
        """
        pass

    def compile(self, meta):
        """Resolves the variable and literal bindings such that
        the expression can be evaluated efficiently later.
        """
        pass

    def ground(self, inst, lbl, meta):
        """Grounds the expression with values from
        the instance and returns a string representation.

        This is useful for debugging since it makes the
        evaluation transparent.
        """
        pass

    def expr(self, meta):
        """Returns a readable string representation of the rule that can be parsed."""

    def get_variables(self):
        """Returns the set of variables bound to the expression"""
        raise TypeError("Unsupported operation: get_variables()")

    def __repr__(self):
        return str(self)


class Term(Expression):
    pass


class Literal(Term):
    """A literal.

    Literals might be numeric, strings, or categorical.
    If categorical, they are converted to internal (integer)
    representation when compiled.

    Attributes:
        val: object (float/string/category)
            The actual value
        valindex: int
            If the corresponding feature is categorical, then
            the integer representation of the value is stored.
        categorical: boolean
            Indicates whether the literal is categorical.

        rexp: Compiled regular expression
            Used to remove quotes (surrounding the literal),
            if required.
    """

    rexp = re.compile(r" *(['\"])(.*)\1 *")

    def __init__(self, val, removequotes=False):
        self.val = val
        self.valindex = UNKNOWN_CATEGORICAL_VALUE_INDEX
        self.categorical = None
        if removequotes:
            m = Literal.rexp.match(val)
            if m is not None:
                self.val = m.group(2)
            else:
                pass

    def evaluate(self, inst, lbl, meta):
        if self.categorical is None:
            raise ValueError("Undetermined whether literal '" + str(self.val) + "' is categorical or not.")
        elif self.categorical:
            ret = self.valindex
        else:
            ret = self.val
        # print(str(self) + ': ' + str(ret))
        return ret

    def ground(self, inst, lbl, meta):
        return repr(self.val)

    def expr(self, meta):
        if meta is None:
            raise ValueError("Invalid metadata")
        if self.categorical is not None and self.categorical:
            return "'" + str(self.val) + "'"
        return str(self.val)

    def get_variables(self):
        return set([])  # A literal does not contain any variable

    def __str__(self):
        return "Lit(" + str(self.val) + "<" + str(self.valindex) + ">" + ")"


class Var(Term):
    """Variable/Feature.

    This represents the feature/label variable. Initially, the
    name of the feature is stored. This gets compiled later into
    the feature index such that evaluation can be faster.

    Label variable is indicated by the feature index '-1'.
    Before compilation, the feature index is initialized to '-2'.
    After compilation, the feature index corresponding to the
    variable name is looked up using the metadata.

    Attributes:
        name: string
            The name of the feature. Usually corresponds to a
            columnname on the data.
        varindex: int
            The index of the variable -- computed during compilation.
        vartype: int (default DTYPE_CATEGORICAL)
            The datatype for the variable.
    """

    def __init__(self, name):
        self.name = name
        self.varindex = UNINITIALIZED_VAR_INDEX  # uninitialized
        self.vartype = DTYPE_CATEGORICAL

    def evaluate(self, inst, lbl, meta):
        ret = None
        if self.varindex == LABEL_VAR_INDEX:
            ret = lbl
        elif self.vartype == DTYPE_CATEGORICAL and self.varindex >= 0:
            # self.vartype is categorical
            ret = int(inst[self.varindex])
        elif self.vartype == DTYPE_CONTINUOUS and self.varindex >= 0:
            # self.vartype is numeric continuous
            ret = inst[self.varindex]
        # print(str(self) + ': ' + str(ret))
        return None if self.vartype == DTYPE_CATEGORICAL and ret < 0 else ret

    def compile(self, meta):
        self.varindex = UNINITIALIZED_VAR_INDEX  # set to uninitialized first
        # print('Compiling Var ' + str(self.name))
        if self.name == meta.lblname:
            self.varindex = LABEL_VAR_INDEX  # label column
        else:
            idx = meta.featurenames.index_of(self.name)
            # -3 = unknown
            self.varindex = ILLEGAL_VAR_INDEX if idx is None else idx
        if self.varindex == ILLEGAL_VAR_INDEX:
            raise ValueError("Unknown variable: '%s' in expression. Allowed variables: %s or '%s'" %
                             (self.name, str(meta.featurenames.all_values().keys()), meta.lblname))
        if self.varindex >= 0 and meta.featuredefs is not None \
                and meta.featuredefs[self.varindex].is_continuous():
            self.vartype = DTYPE_CONTINUOUS  # DTYPE_CONTINUOUS
        else:
            self.vartype = DTYPE_CATEGORICAL  # DTYPE_CATEGORICAL

    def ground(self, inst, lbl, meta):
        val = "?"
        if self.varindex == LABEL_VAR_INDEX:  # label
            val = "'" + meta.lbldef[lbl] + "'"
        elif self.varindex >= 0:
            # assume that all features are continuous
            val = inst[self.varindex]
        return str(self.name) + "(" + repr(val) + ")"

    def expr(self, meta):
        if self.varindex == LABEL_VAR_INDEX:  # label
            return meta.lblname
        elif self.varindex >= 0:
            return meta.featurenames[self.varindex]
        raise ValueError("Uncompiled Rule: %s" % (str(self),))

    def get_variables(self):
        return {self.varindex}

    def __str__(self):
        return "Var(" + str(self.name) + "<" + str(self.varindex) + ">)"


class Predicate(Expression):
    pass


class Atom(Predicate):
    pass


class BinaryPredicate(Predicate):
    """Predicate taking two inputs."""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        self.p1 = p1
        self.p2 = p2
        self.weight = weight

    def compile(self, meta):
        # print('Compiling ' + str(self.p1) + ' ' + str(isinstance(self.p1, Predicate)))
        self.p1.compile(meta)
        # print('Compiling ' + str(self.p2) + ' ' + str(isinstance(self.p2, Predicate)))
        self.p2.compile(meta)

    def get_variables(self):
        vars = set()
        vars.update(self.p1.get_variables())
        vars.update(self.p2.get_variables())
        return vars

    def get_str_weight(self, suppress_default_weight=True):
        if suppress_default_weight and self.weight == DEFAULT_PREDICATE_WEIGHT:
            return ""
        return "[" + str(self.weight) + "]"


class UnaryPredicate(Predicate):
    """Predicate taking one input."""

    def __init__(self, p=None, weight=DEFAULT_PREDICATE_WEIGHT):
        self.p = p
        self.weight = weight

    def compile(self, meta):
        self.p.compile(meta)

    def get_variables(self):
        vars = set()
        vars.update(self.p.get_variables())
        return vars

    def get_str_weight(self, suppress_default_weight=True):
        if suppress_default_weight and self.weight == DEFAULT_PREDICATE_WEIGHT:
            return ""
        return "[" + str(self.weight) + "]"


class Cmp(BinaryPredicate):
    """Base class for evaluating comparison operators."""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        BinaryPredicate.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluate(self, inst, lbl, meta):
        e1 = self.p1.evaluate(inst, lbl, meta)
        e2 = self.p2.evaluate(inst, lbl, meta)
        ret = None if e1 is None or e2 is None else self.evaluateCmp(e1, e2)
        # print(str(self) + ': ' + str(ret))
        if ret is None:
            raise ValueError('predicate value for %s unbound \n inst: %s' \
                             % (str(self), str(inst)))
        return ret

    def evaluateCmp(self, e1, e2):
        raise NotImplementedError('Comparison operator not implemented.')

    def compile(self, meta):
        self.p1.compile(meta)
        self.p2.compile(meta)
        # Comparisons must be between a variable and a literal.
        tvar = self.p1 if isinstance(self.p1, Var) else self.p2 if isinstance(self.p2, Var) else None
        tlit = self.p1 if isinstance(self.p1, Literal) else self.p2 if isinstance(self.p2, Literal) else None
        if tvar is not None and tlit is not None:
            if tvar.varindex == LABEL_VAR_INDEX:  # label column
                tlit.categorical = True  # class variables are always categorical
                tlit.valindex = meta.lbldef.index_of(tlit.val)
            elif tvar.varindex >= 0:  # feature column
                if isinstance(meta.featuredefs[tvar.varindex], Factor):
                    tlit.categorical = True
                    valindex = meta.featuredefs[tvar.varindex].index_of(tlit.val)
                    if valindex is None:
                        valindex = UNKNOWN_CATEGORICAL_VALUE_INDEX
                    tlit.valindex = valindex
                else:
                    tlit.categorical = False
        else:
            raise ValueError('Comparisons must be between a variable and a literal.')

    def ground(self, inst, lbl, meta):
        raise NotImplementedError('ground() not implemented.')

    def __str__(self):
        return "Cmp(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class CmpEq(Cmp):
    """Compares if values of two expressions are equal"""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        Cmp.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluateCmp(self, e1, e2):
        return e1 == e2

    def ground(self, inst, lbl, meta):
        return "" + self.p1.ground(inst, lbl, meta) + " = " + self.p2.ground(inst, lbl, meta) + ""

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " = " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "CmpEq(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class CmpLr(Cmp):
    """Compares if e1 < e2"""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        Cmp.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluateCmp(self, e1, e2):
        return e1 < e2

    def ground(self, inst, lbl, meta):
        return "" + self.p1.ground(inst, lbl, meta) + " < " + self.p2.ground(inst, lbl, meta) + ""

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " < " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "CmpLr(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class CmpLE(Cmp):
    """Compares if e1 <= e2"""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        Cmp.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluateCmp(self, e1, e2):
        return e1 <= e2

    def ground(self, inst, lbl, meta):
        return "" + self.p1.ground(inst, lbl, meta) + " <= " + self.p2.ground(inst, lbl, meta) + ""

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " <= " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "CmpLE(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class CmpGr(Cmp):
    """Compares if e1 > e2"""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        Cmp.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluateCmp(self, e1, e2):
        return e1 > e2

    def ground(self, inst, lbl, meta):
        return "" + self.p1.ground(inst, lbl, meta) + " > " + self.p2.ground(inst, lbl, meta) + ""

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " > " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "CmpGr(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class CmpGE(Cmp):
    """Compares if e1 >= e2"""

    def __init__(self, p1=None, p2=None, weight=DEFAULT_PREDICATE_WEIGHT):
        Cmp.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluateCmp(self, e1, e2):
        return e1 >= e2

    def ground(self, inst, lbl, meta):
        return "" + self.p1.ground(inst, lbl, meta) + " >= " + self.p2.ground(inst, lbl, meta) + ""

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " >= " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "CmpGE(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class Or(BinaryPredicate):
    def __init__(self, p1, p2, weight=DEFAULT_PREDICATE_WEIGHT):
        BinaryPredicate.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluate(self, inst, lbl, meta):
        e1 = self.p1.evaluate(inst, lbl, meta)
        if e1 is None:
            raise ValueError('predicate value unbound for e1')
        elif e1:
            return True
        e2 = self.p2.evaluate(inst, lbl, meta)
        ret = None if e1 is None or e2 is None else e1 or e2
        # print(str(self) + ': ' + str(ret))
        if ret is None:
            raise ValueError('predicate value unbound for e2')
        return ret

    def ground(self, inst, lbl, meta):
        return "(" + self.p1.ground(inst, lbl, meta) + " | " + self.p2.ground(inst, lbl, meta) + ")"

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " | " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "Or(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class And(BinaryPredicate):
    def __init__(self, p1, p2, weight=DEFAULT_PREDICATE_WEIGHT):
        BinaryPredicate.__init__(self, p1=p1, p2=p2, weight=weight)

    def evaluate(self, inst, lbl, meta):
        e1 = self.p1.evaluate(inst, lbl, meta)
        if e1 is None:
            raise ValueError('predicate value unbound for e1')
        elif not e1:
            return False
        e2 = self.p2.evaluate(inst, lbl, meta)
        ret = None if e1 is None or e2 is None else e1 and e2
        # print(str(self) + ': ' + str(ret))
        if ret is None:
            raise ValueError('predicate value unbound for e2')
        return ret

    def ground(self, inst, lbl, meta):
        return "(" + self.p1.ground(inst, lbl, meta) + " & " + self.p2.ground(inst, lbl, meta) + ")"

    def expr(self, meta):
        return "(" + self.p1.expr(meta) + " & " + self.p2.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "And(" + str(self.p1) + ", " + str(self.p2) + ")" + self.get_str_weight()


class Not(UnaryPredicate):
    def __init__(self, p, weight=DEFAULT_PREDICATE_WEIGHT):
        UnaryPredicate.__init__(self, p=p, weight=weight)

    def evaluate(self, inst, lbl, meta):
        e = self.p.evaluate(inst, lbl, meta)
        ret = None if e is None else not e
        # print(str(self) + ': ' + str(ret))
        if ret is None:
            raise ValueError('predicate value unbound')
        return ret

    def ground(self, inst, lbl, meta):
        return "~(" + self.p.ground(inst, lbl, meta) + ")"

    def expr(self, meta):
        return "~(" + self.p.expr(meta) + ")" + self.get_str_weight()

    def __str__(self):
        return "Not(" + str(self.p) + ")" + self.get_str_weight()


class RuleParser(object):
    """Methods to parse strings as Expression objects."""

    # noinspection PyMethodMayBeStatic
    def parse(self, s):
        """Parses string 's' and returns an Expression object.

        :param s: str
        :return: Predicate
        """

        # A kludgy way to make it work for both Python 2.7 and 3.5+
        try:
            import StringIO
            rdr = StringIO.StringIO(s).readline
        except:
            from io import StringIO
            rdr = StringIO(s).readline

        def precedence(op):
            """Higher value means higher precedence"""
            if op == "|":
                return 1
            elif op == "&":
                return 2
            elif op == "~":
                return 3
            elif op == "=" or op == "<=" or op == "<" or op == ">" or op == ">=":
                return 4
            elif op == "":  # usually as endmarker
                return 0
            else:
                return 0

        def consume_operator(astk, ostk, op):
            while not ostk.is_empty():
                top = ostk[len(ostk) - 1]
                # print("top: %s op: %s precedence(top): %d precedence(op): %d" % (top,op,precedence(top),precedence(op)))
                if op == ")" and top == "(":
                    ostk.pop()
                    break
                elif op == "]" and top == "[":
                    # populate predicate weight
                    ostk.pop()
                    # There must be a predicate and a numeric literal on stack
                    if len(astk) < 2:
                        raise ValueError("invalid weight found")
                    wtlit = astk.pop()
                    pred = astk.pop()
                    if not isinstance(wtlit, Literal) or not isinstance(pred, Predicate):
                        raise ValueError("invalid weight format")
                    pred.weight = wtlit.val
                    astk.push(pred)
                    break
                elif op == "]" and not top == "[":
                    raise ValueError("invalid ']' found")
                if precedence(op) <= precedence(top):
                    if top == "=":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(CmpEq(t1, t2))
                    elif top == "<":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(CmpLr(t1, t2))
                    elif top == "<=":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(CmpLE(t1, t2))
                    elif top == ">":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(CmpGr(t1, t2))
                    elif top == ">=":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(CmpGE(t1, t2))
                    elif top == "~":
                        ostk.pop()
                        t1 = astk.pop()
                        astk.push(Not(t1))
                    elif top == "&":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(And(t1, t2))
                    elif top == "|":
                        ostk.pop()
                        t2 = astk.pop()
                        t1 = astk.pop()
                        astk.push(Or(t1, t2))
                    else:
                        break
                else:
                    break

        astk = stack()
        ostk = stack()  # operator stack
        g = tokenize.generate_tokens(rdr)  # tokenize the string
        ret = None
        for toknum, tokval, _, _, _ in g:
            if toknum == tokenize.OP:
                # print('OP ' + tokval + ' ' + str(toknum))
                if tokval == "(":  # nested predicate
                    ostk.push(tokval)
                elif tokval == ")":
                    consume_operator(astk, ostk, tokval)
                elif tokval == "[":  # predicate weight
                    ostk.push(tokval)
                elif tokval == "]":
                    consume_operator(astk, ostk, tokval)
                elif tokval == "-":  # handle negative numbers
                    ostk.push(tokval)
                elif tokval in ["=", "&", "|", "~", "<=", "<", ">", ">="]:
                    consume_operator(astk, ostk, tokval)
                    ostk.push(tokval)
                else:
                    raise SyntaxError("Illegal operator '" + tokval + "' found in rule expression")
            elif toknum == tokenize.NAME:
                # print('NAME ' + tokval + ' ' + str(toknum))
                astk.push(Var(tokval))
            elif toknum == tokenize.STRING:
                # print('STR/NUM ' + tokval + ' ' + str(toknum))
                astk.push(Literal(tokval, removequotes=True))
            elif toknum == tokenize.NUMBER:
                # print('STR/NUM ' + tokval + ' ' + str(toknum))
                sign = 1
                if len(ostk) > 0 and ostk[len(ostk) - 1] == "-":
                    sign = -1
                    ostk.pop()
                astk.push(Literal(sign * float(tokval)))
            elif toknum == tokenize.INDENT or toknum == tokenize.DEDENT:
                pass
            elif toknum == tokenize.ENDMARKER:
                consume_operator(astk, ostk, "")
                ret = None if astk.is_empty() else astk.pop()
                # print(ret)
                if not astk.is_empty():
                    print(astk)
                    print(ostk)
                    raise SyntaxError("Invalid rule syntax in " + str(s))
            else:
                print('UNK ' + tokval + ' ' + str(toknum))
            # print("astk: %s" % (str(astk),))
            # print("ostk: %s" % (str(ostk),))
        return ret


def string_to_predicate(str_predicate, meta=None, parser=None):
    """Converts a string representation of rule to a Predicate object"""

    parser_ = RuleParser() if parser is None else parser
    predicate = parser_.parse(str_predicate)

    if meta is not None:
        predicate.compile(meta)

    return predicate


class PredicateContext(object):
    """Holds predicate traversal context

    Attributes:
        neg: boolean
        features: set of tuples
    """

    def __init__(self):
        self.neg = False
        self.features = []


def traverse_predicate_conjunctions(predicate, context):
    """ Updates traversal context recursively

    Collects all the And/Cmp predicate expressions into predicate list.
    Expressions other than {Cmp, And} will raise error.

    :param predicate: Expression
    :param context: PredicateContext
    :return: None
    """

    if isinstance(predicate, Cmp):
        p1 = None  # holds Var
        p2 = None  # holds Literal

        if isinstance(predicate.p1, Var):
            p1 = predicate.p1
        elif isinstance(predicate.p2, Var):
            p1 = predicate.p2

        if isinstance(predicate.p1, Literal):
            p2 = predicate.p1
        elif isinstance(predicate.p2, Literal):
            p2 = predicate.p2

        if p1 is not None and p2 is not None:
            context.features.append((p1, p2, predicate))
        else:
            raise ValueError("Unbound Var or Literal. Expected Comparison of Var and Literal.")

    elif isinstance(predicate, And):
        traverse_predicate_conjunctions(predicate.p1, context)
        traverse_predicate_conjunctions(predicate.p2, context)

    else:
        raise ValueError("Expected conjunctive form, but found Or()")

    return context


def conjunctive_predicate_to_list(predicate):
    context = PredicateContext()
    traverse_predicate_conjunctions(predicate, context)
    return context.features


class ConjunctiveRule(object):
    """ Represents a conjunction (And) of simple one-feature-value comparison predicates """

    def __init__(self, predicates, meta, id=None):
        self.predicates = predicates
        self.meta = meta
        # id might be required e.g., when we have to remember the node
        # of the tree that this corresponds to.
        self.id = id
        self.support = None
        self.confusion_matrix = None

    def set_confusion_matrix(self, positive_indexes, y):
        mask = np.array([True] * len(y))
        mask[positive_indexes] = False
        negative_indexes = np.where(mask)[0]
        tp = np.sum(y[positive_indexes])
        fp = len(positive_indexes) - tp
        tn = np.sum(y[negative_indexes])
        fn = len(negative_indexes) - tn
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]], dtype=np.float32)
        self.support = tp * 1.0 / (tp + fp)

    @staticmethod
    def parse(str_rule, meta):
        rule = string_to_predicate(str_rule, meta)

        conjunctions = conjunctive_predicate_to_list(rule)

        predicates = []
        for p1, p2, predicate in conjunctions:
            if not (isinstance(predicate.p1, Var) and isinstance(predicate.p2, Literal)):
                raise ValueError("Conjunctive predicates must be of format: Variable = Literal")
            predicates.append(predicate)
        return ConjunctiveRule(predicates, meta)

    def evaluate_inst(self, inst, label):
        """ Checks if the instance satisfies all the predicates (i.e., 'And') """
        result = True
        i = 0
        while result and i < len(self.predicates):
            result = result and self.predicates[i].evaluate(inst, label, self.meta)
            i += 1
        return result

    def where_satisfied(self, insts, labels=None):
        """ Returns all indexes of insts which satisfy the rule

        :param insts: np.ndarray
        :param labels: np.array
        :return: np.array
        """
        satisfied = []
        for i in range(insts.shape[0]):
            if self.evaluate_inst(insts[i, :], None if labels is None else labels[i]):
                satisfied.append(i)
        return np.array(satisfied, dtype=np.int32)

    def _str_confusion_mat(self):
        if self.confusion_matrix is None:
            return 'None'
        else:
            return "[%s, %s]" % \
                   (str(list(self.confusion_matrix[0,:])), str(list(self.confusion_matrix[1,:])))

    def __str__(self):
        predicate_strs = []
        for predicate in self.predicates:
            predicate_strs.append(predicate.expr(self.meta))
        return " & ".join(predicate_strs)

    def __len__(self):
        if self.predicates is not None:
            return len(self.predicates)
        return 0

    def __repr__(self):
        predicate_strs = []
        for predicate in self.predicates:
            predicate_strs.append(predicate.expr(self.meta))
        return "%s%s%s" % \
               ("" if self.support is None else "support: %0.4f; " % self.support,
                " & ".join(predicate_strs),
                "" if self.confusion_matrix is None else "; %s" % self._str_confusion_mat())


def convert_feature_ranges_to_rules(ranges, meta):
    """ Converts list of maps of feature-ranges to Rule objects.

    Each range map in the input list will be converted to a separate Rule.

    The leaf nodes of a tree-based model usually partition the feature
    space into subspaces defined by corresponding feature ranges. These
    feature-ranges can be represented by the ConjunctiveRule data structure.

    :param ranges: list of dict
        [{feature_index: (min_val, max_val), ...}, ...]
    :param meta: FeatureMetadata
    :return: list of ConjunctiveRule, list of strings
    """
    rules = []
    str_rules = []
    for range_map in ranges:
        predicates = []
        for feature, range in range_map.items():
            if np.isfinite(range[0]):
                predicates.append("%s > %f" % (meta.featurenames[feature], range[0]))
            if np.isfinite(range[1]):
                predicates.append("%s <= %f" % (meta.featurenames[feature], range[1]))
        if len(predicates) > 0:
            str_rule = " & ".join(predicates)
            rules.append(ConjunctiveRule.parse(str_rule, meta))
            str_rules.append(str_rule)
    return rules, str_rules


def convert_conjunctive_rule_to_feature_ranges(rule, meta):
    extents = dict()
    for feature, featuredef in enumerate(meta.featuredefs):
        if featuredef.is_continuous():
            extents[feature] = (-np.inf, np.inf)
    for predicate in rule.predicates:
        feature = predicate.p1.varindex
        if not meta.featuredefs[feature].is_continuous():
            continue
        value = predicate.p2.val
        f_range = extents[feature]
        if isinstance(predicate, CmpGr) or isinstance(predicate, CmpGE):
            f_range = (max(f_range[0], value), f_range[1])
        elif isinstance(predicate, CmpLr) or isinstance(predicate, CmpLE):
            f_range = (f_range[0], min(f_range[1], value))
        extents[feature] = f_range
    return extents


def convert_conjunctive_rules_to_strings(rules):
    return [str(rule) for rule in rules]


def convert_conjunctive_rules_to_feature_ranges(rules, meta):
    ranges = [convert_conjunctive_rule_to_feature_ranges(rule, meta) for rule in rules]
    return ranges


def convert_strings_to_conjunctive_rules(str_rules, meta):
    rules = []
    for str_rule in str_rules:
        rules.append(ConjunctiveRule.parse(str_rule, meta))
    return rules


def get_max_len_in_rules(rules):
    return max([len(rule) for rule in rules])


def get_rule_satisfaction_matrix(x, y, rules):
    """ Returns a matrix that shows which instances satisfy which rules

    Each column of the returned matrix corresponds to a rules and each row to an instance.
    If an instance satisfies a rule, the corresponding value will be 1, else 0.

    :param x: np.ndarray
    :param y: np.array
    :param rules: list
    :param opts: AadOpts
    :return: np.ndarray
        matrix with x.shape[0] rows and len(rules) rows
    """
    satisfaction_matrix = np.zeros((x.shape[0], len(rules)), dtype=np.int32)
    for i, rule in enumerate(rules):
        idxs = rule.where_satisfied(x, y)
        satisfaction_matrix[idxs, i] = 1
    return satisfaction_matrix


def check_if_at_least_one_rule_satisfied(x, y, rules):
    """ For each input instance, check if it satisfies at least one rule

    Basically performs a disjunction of rules.
    Can be applied to rules in DNF format.

    :param x: np.ndarray
    :param y: np.array
        This could be None if unsupervised and if it is not required to evaluate any rule
    :param rules: list of rules
    :return: np.array
        Binary indicator for each instance
    """
    sat_vec = np.zeros(x.shape[0], dtype=np.int32)
    for rule in rules:
        idxs = rule.where_satisfied(x, y)
        sat_vec[idxs] += 1
    return np.minimum(sat_vec, 1)


def evaluate_ruleset(x, y, rules, average="binary"):
    """ For each input instance, check if it satisfies at least one rule and computes F1 """
    y_hat = check_if_at_least_one_rule_satisfied(x, y, rules)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y, y_pred=y_hat, average=average)
    return precision, recall, f1


def save_strings_to_file(strs, file_path):
    if file_path is None or file_path == '':
        raise ValueError
    with open(file_path, 'w') as f:
        for s in strs:
            f.write(s + os.linesep)


def load_strings_from_file(file_path):
    strs = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line != "":
                strs.append(line)
    return strs


def get_feature_meta_default(x, y, feature_names=None,
                             label_name='label', labels=None, featuredefs=None):
    """ A simple convenience method that creates a default FeatureMetadata

    In the default metadata:
      1. If feature names are not provided, the columns/features of x
         are assigned names F1, F2, ...
      2. The class label is referred to as 'label'
      3. All columns are treated as continuous numeric

    In case dataset-specific names are to be assigned, create the appropriate
    metadata in a similar manner as illustrated here.
    """
    if feature_names is None:
        f_names = Factor(["F%d" % (i+1) for i in range(x.shape[1])], sort=False)
    else:
        if x.shape[1] != len(feature_names):
            raise ValueError("feature_names should have same length as columns in x")
        f_names = Factor(feature_names, sort=False)
    if featuredefs is None:
        featuredefs = [NumericContinuous(x[:, i]) for i in range(x.shape[1])]
    if labels is None:
        labels = np.unique(y)
    meta = FeatureMetadata(lblname=label_name, lbldef=Factor(labels),
                           featurenames=f_names, featuredefs=featuredefs)
    return meta


def evaluate_instances_for_predicate(predicate, insts, labels, meta):
    satisfied = []
    for i in range(insts.shape[0]):
        lbl = labels[i] if labels is not None else None
        if predicate.evaluate(insts[i, :], lbl, meta):
            satisfied.append(i)
    return np.array(satisfied, dtype=np.int32)


def test_rule_apis():
    from .gen_samples import read_anomaly_dataset
    x, y = read_anomaly_dataset("toy2")
    y = np.asarray(y, dtype=np.int32)

    meta = get_feature_meta_default(x, y)
    print(meta)

    parser = RuleParser()

    """
    Since the default metadata names the features as F1, F2, ... and
    the class label as 'label', we will refer to these by the same names
    in the predicate rules.
    
    RuleParser can parse any well-formed logical predicates such as
    those in predicate_strs below.
    """

    predicate_strs = [
        # internal representation:
        #   CmpEq(Var(label<-1>), Lit(0.0<0>))
        # Var(label<-1>) : <-1> means that the variable 'label' is not a regular feature
        # Lit(0.0<0>) : the label '0' got changed to 0.0 because it was numeric.
        #     To make label '0', change the label to string.
        #     <0> means that '0' is at the 0-th position of the label Factor
        "label = 0",  # all 0 labeled

        # internal representation:
        #   CmpEq(Var(label<-1>), Lit(1.0<1>))
        "label = 1",  # all 1 labeled

        # internal representation:
        #   And(Or(Or(CmpGE(Var(F1<0>), Lit(0.0<-1>)), CmpLr(Var(F2<1>), Lit(2.0<-1>))), CmpLr(Var(F1<0>), Lit(-5.0<-1>))), CmpGr(Var(F2<1>), Lit(0.0<-1>)))
        # Var(F1<0>) : feature 'F1' is the 0-th feature
        # Var(F2<1>) : feature 'F2' is the 1-st feature
        # Lit(0.0<-1>) : <-1> here means 0.0 is numeric, and not categorical
        # ... and so on ...
        "(F1 >= 0 | F2 < 2 | F1 < -5) & F2 > 0",  # just an arbitrary predicate

        # internal representation:
        #   Or(Not(CmpGE(Var(F2<1>), Lit(2.0<-1>))), CmpEq(Var(label<-1>), Lit(1.0<1>)))
        "(~(F2 >= 2) | (label = 1))",  # a Horn clause: (F2 >= 2) => (label = 1)

        # internal representation:
        #   And(And(And(CmpGE(Var(F1<0>), Lit(1.0<-1>)), CmpLr(Var(F1<0>), Lit(5.0<-1>))), CmpGE(Var(F2<1>), Lit(0.0<-1>))), CmpLr(Var(F2<1>), Lit(6.0<-1>)))
        "F1 >= 1 & F1 < 5 & (F2 >= 0) & (F2 < 6)",  # conjunctive predicate
    ]

    for predicate_str in predicate_strs:
        predicate = parser.parse(predicate_str)
        predicate.compile(meta)  # bind feature indexes to feature names
        matches = evaluate_instances_for_predicate(predicate, x, y, meta)
        print("%s matched: %d\n  repr: %s" % (predicate.expr(meta), len(matches), str(predicate)))

    # the rule(s) below are conjunctive
    # conjunctive_str = "(F1 >= 1) & (F1 < 5) & (F2 >= 0) & (F2 < 6)"
    conjunctive_str = "F1 >= 1 & F1 < 5 & (F2 >= 0) & (F2 < 6)"

    # conjunctive rules can be used with the convenience class ConjunctiveRule
    rule = ConjunctiveRule.parse(conjunctive_str, meta)
    idxs = rule.where_satisfied(x, y)
    rule.set_confusion_matrix(idxs, y)
    print(str(rule))


if __name__ == "__main__":
    test_rule_apis()
