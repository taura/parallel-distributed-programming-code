#!/usr/bin/python3
"""
parse_log
"""
import csv
import json
import re
import sys
import time
#import pdb

class parse_error(Exception):
    """
    parse error class
    """

class log_parser_base:
    """
    base class for log parsers
    """
    def __init__(self, fp):
        self.fp = fp
        tokens = {
            "open_log"          : r"open a log (?P<when>.+)",
            "close_log"         : r"close a log (?P<when>.+)",
            "env"               : r"(?P<var>[A-Za-z0-9_\-]+)( undefined|=(?P<val>.+))",
            "model_start"       : r"model building starts",
            "model_end"         : r"model building ends",
            "load_start"        : r"loading data from (?P<data>.+)",
            "load_end"          : r"use (?P<n>\d+) data items out of \d+",
            "training_start"    : r"training starts",
            "training_end"      : r"training ends",
            "train_epoch_start" : r"Train Epoch (?P<epoch>\d+) starts",
            "test_epoch_start"  : r"Test Epoch (?P<epoch>\d+) starts",
            "train_epoch_end"   : r"Train Epoch (?P<epoch>\d+) ends",
            "test_epoch_end"    : r"Test Epoch (?P<epoch>\d+) ends",
            "train_batch_start" : (r"Train Epoch (?P<epoch>\d+) batch (?P<batch>\d+)"
                                   " \(samples (?P<a>\d+) - (?P<b>\d+)\) starts"),
            "train_batch_end"   : (r"Train Epoch (?P<epoch>\d+) batch (?P<batch>\d+)"
                                   " \(samples (?P<a>\d+) - (?P<b>\d+)\) ends"),
            "test_batch_start"  : (r"Test Epoch (?P<epoch>\d+) batch (?P<batch>\d+)"
                                   " \(samples (?P<a>\d+) - (?P<b>\d+)\) starts"),
            "test_batch_end"    : (r"Test Epoch (?P<epoch>\d+) batch (?P<batch>\d+)"
                                   " \(samples (?P<a>\d+) - (?P<b>\d+)\) ends"),
            "train_loss"        : (r"Train Epoch: (?P<epoch>\d+) \[(?P<a>\d+)/(?P<n>\d+) \((?P<p>\d+)%\)\]"
                                   "\s+Loss: (?P<loss>\d+\.\d+)"),
            "test_loss"         : r"Test set: Average loss: (?P<loss>\d+\.\d+), Accuracy: (?P<c>\d+)/(?P<n>\d+) \((?P<p>\d+)%\)",
            "sample"            : (r"sample (?P<sample>\d+) image (?P<image>\d+)"
                                   r" pred (?P<pred>\d+) truth (?P<truth>\d+)"),
            "kernel_start"      : r"(?P<kernel>.*): starts",
            "kernel_end"        : r"(?P<kernel>.*): ends\. took (?P<kernel_time>\d+) nsec",
        }
        pat_time = r"(?P<t>\d+): "
        self.patterns = {tok : re.compile(pat_time + pat) for tok, pat in tokens.items()}
        self.patterns["EOF"] = re.compile("^$")
        self.kpsr = kernel_parser()
        self.lines = []
        self.next_line()

    def next_line(self):
        """
        get next line, set token kind
        """
        self.line = self.fp.readline()
        if self.line != "":
            #print(self.line.strip())
            self.lines.append(self.line)
        for tok, regex in self.patterns.items():
            match = regex.match(self.line)
            if match:
                self.tok = tok
                self.data = match.groupdict()
                # print("  [{}] {}".format(self.tok, self.data))
                return
        self.tok = None
        self.data = None
    def call_action(self, tok):
        """
        call callback defined in subclasses
        """
        method_name = "action_%s" % tok
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method(self.data)
    def eat(self, tok):
        """
        make sure current token type is tok and
        get next token
        """
        if self.tok != tok:
            self.parse_error(tok)
        self.call_action(tok)
        self.next_line()
    def parse_error(self, tok):
        """
        raise parse error
        """
        raise parse_error("%s:%d:error: expected %s but got %s\n[%s]\n" %
                          (self.fp.name, len(self.lines), tok, self.tok, self.line))
    def parse_kernel_start(self):
        """
        ...: starts
        """
        self.kpsr.parse(self.data["kernel"])
        self.eat("kernel_start")
    def parse_kernel_end(self):
        """
        ...: ends. took ... nsec
        """
        self.kpsr.parse(self.data["kernel"])
        self.eat("kernel_end")
    def parse_train_batch(self):
        """
        Train Epoch AAA batch BBB (samples CCC - DDD) starts
        [Train Epoch: AAA [BBB/CCC (1%)]	Loss: F.FFFF]?
        Train Epoch AAA batch BBB (samples CCC - DDD) ends
        """
        self.eat("train_batch_start")
        while self.tok == "kernel_start":
            self.parse_kernel_start()
            self.parse_kernel_end()
        while self.tok == "sample":
            self.eat("sample")
        if self.tok == "train_loss":
            self.eat("train_loss")
        self.eat("train_batch_end")
    def parse_test_batch(self):
        """
        Test Epoch AAA batch BBB (samples CCC - DDD) starts
        [Test Epoch: AAA [BBB/CCC (1%)]	Loss: F.FFFF]?
        Test Epoch AAA batch BBB (samples CCC - DDD) ends
        """
        self.eat("test_batch_start")
        while self.tok == "kernel_start":
            self.parse_kernel_start()
            self.parse_kernel_end()
        # it should not be here, but mnist.cc prints batch end before samples
        if self.tok == "test_batch_end":
            self.eat("test_batch_end")
        while self.tok == "sample":
            self.eat("sample")
        # it should be here
        if self.tok == "test_batch_end":
            self.eat("test_batch_end")
        if self.tok == "test_loss":
            self.eat("test_loss")
    def parse_train_epoch(self):
        """
        227997219: Train Epoch x starts
          228023464: Train Epoch x batch y (samples a - b) starts
          ...
          228023464: Train Epoch x batch y (samples a - b) ends
        227997219: Train Epoch x ends
        """
        self.eat("train_epoch_start")
        while self.tok == "train_batch_start":
            self.parse_train_batch()
        self.eat("train_epoch_end")
    def parse_test_epoch(self):
        """
        227997219: Train Epoch x starts
          228023464: Train Epoch x batch y (samples a - b) starts
          ...
          228023464: Train Epoch x batch y (samples a - b) ends
        227997219: Train Epoch x ends
        """
        self.eat("test_epoch_start")
        while self.tok == "test_batch_start":
            self.parse_test_batch()
        self.eat("test_epoch_end")
    def parse_validate(self):
        """
        === validate 448 - 512 ===
        ...
        """
        self.eat("validate_begin")
        while self.tok == "kernel_start":
            self.parse_mini_batch()
        self.eat("validate_accuracy")
        self.eat("validate_loss")
    def parse_file(self):
        """
        other* train_or_validate* EOF
        """
        self.eat("open_log")
        while self.tok == "env":
            self.eat("env")
        self.eat("model_start")
        self.eat("model_end")
        self.eat("load_start")
        self.eat("load_end")
        self.eat("load_start")
        self.eat("load_end")
        self.eat("training_start")
        while self.tok == "train_epoch_start":
            self.parse_train_epoch()
            self.parse_test_epoch()
        self.eat("training_end")
        self.eat("close_log")
        self.eat("EOF")

class kernel_parser:
    """
    parse kernel name like
    array4<maxB, OC, ..>&
    Convolution2D<maxB, IC, ..>::forward(array4<maxB, ...>&)
    [with int maxB = 64; ...]
    """
    def __init__(self):
        self.keywords = ["with"]
        patterns_ = [
            ("id", "[A-Za-z_][0-9A-Za-z_]*"),
            ("num", "[1-9][0-9]*"),
            ("<", "<"),
            (",", ","),
            (">", ">"),
            ("&", "&"),
            ("::", "::"),
            ("(", r"\("),
            (")", r"\)"),
            ("[", r"\["),
            ("]", r"\]"),
            ("=", "="),
            ("+", r"\+"),
            ("-", r"\-"),
            ("*", r"\*"),
            ("/", r"\/"),
            ("%", "%"),
            (";", ";"),
        ]
        self.patterns = {(k, re.compile(v)) for k, v in patterns_}
        self.tokenize_pattern = re.compile("(%s)" % "|".join([v for k, v in patterns_]))
        self.first_type = set(["id"])
        self.first_multiplicative = set(["id", "num", "("])
        self.tokens = None
        self.idx = None
        self.tok = None
        self.kind = None
    def init(self, s):
        """
        initialization
        """
        self.tokens = self.tokenize_pattern.findall(s)
        self.idx = -1
        self.next_token()
    def next_token(self):
        """
        next token
        """
        self.idx += 1
        if self.idx < len(self.tokens):
            self.tok = self.tokens[self.idx]
            self.kind = self.token_kind(self.tok)
        else:
            self.tok = ""
            self.kind = "EOF"
        return self.kind
    def token_kind(self, tok):
        """
        kind of tok
        """
        for kind, pattern in self.patterns:
            if pattern.match(tok):
                if tok in self.keywords:
                    return tok
                return kind
        assert(0), tok
        return None
    def eat(self, kinds):
        """
        eat current token
        """
        assert(isinstance(kinds, type([]))), kinds
        for kind in kinds:
            if self.kind == kind:
                tok = self.tok
                self.next_token()
                return tok
        self.parse_error(kinds)
        return None
    def parse_error(self, kinds):
        """
        raise parse error
        """
        raise parse_error("error: expected %s but got '%s' (%s)"
                          % (kinds, self.kind, self.tok))
    def parse_template_expr(self):
        """
        expr ::= multiplicative ( +/- expr )*
        multiplicative ::= primary (*// multiplicative)*
        primary ::= id | num | ( expr )
        """
        if self.kind in self.first_multiplicative:
            expr = self.parse_multiplicative()
            while self.kind in ["+", "-"]:
                operator = self.eat([self.kind])
                expr = (operator, expr, self.parse_template_expr())
        else:
            #nvc++ emits tensor<float, maxB, OC, <expression>, <expression>> 
            self.eat(["<"])
            expr = self.eat(["id"])    # expression
            self.eat([">"])
        return expr
    def parse_multiplicative(self):
        """
        multiplicative ::= primary (*// multiplicative)*
        primary ::= id | num | ( expr )
        """
        expr = self.parse_primary()
        while self.kind in ["*", "/"]:
            operator = self.eat([self.kind])
            expr = (operator, expr, self.parse_multiplicative())
        return expr
    def parse_primary(self):
        """
        primary ::= id | num | ( expr )
        """
        if self.kind == "(":
            self.eat(["("])
            expr = self.parse_template_expr()
            self.eat([")"])
            return expr
        if self.kind == "id":
            return self.eat(["id"])
        if self.kind == "num":
            return int(self.eat(["num"]))
        raise parse_error("primary expected (, id, or num, got %s"
                          % self.kind)
    def parse_id(self):
        """
        var<expr, expr, ...>
        """
        name = self.eat(["id"])
        if self.kind == "<":
            args = []
            self.eat(["<"])
            if self.kind in ["id", "num"]:
                args.append(self.parse_template_expr())
                while self.kind == ",":
                    self.eat([","])
                    args.append(self.parse_template_expr())
            self.eat([">"])
        else:
            args = None
        return dict(name=name, args=args)
    def parse_type(self):
        """
        var<expr, expr, ...>[&]
        """
        dic = self.parse_id()
        name = dic["name"]
        args = dic["args"]
        if self.kind == "&":
            amp = self.eat(["&"])
        else:
            amp = None
        return dict(name=name, args=args, amp=amp)
    def parse_class_fun_name(self):
        """
        var<expr, expr, ...>::var
        """
        dic = self.parse_id()
        if self.kind == "::":
            self.eat(["::"])
            class_name = dic["name"]
            class_args = dic["args"]
            fun = self.parse_id()
            fun_name = fun["name"]
            fun_args = fun["args"]
        else:
            class_name = None
            class_args = None
            fun_name = dic["name"]
            fun_args = dic["args"]
        return dict(class_name=class_name, class_args=class_args,
                    fun_name=fun_name, fun_args=fun_args)
    def parse_instantiation(self):
        """
        type id = expr | id = type_expr
        """
        tok0 = self.eat(["id"])
        if self.kind == "id":
            # int x  = 10
            var = self.eat(["id"])
            self.eat(["="])
            expr = self.parse_template_expr()
            return (var, ("val", expr))
        if self.kind == "=":
            # e.g., real = float | var = num
            var = tok0
            self.eat(["="])
            if self.kind in self.first_type:
                # real = float
                expr = self.parse_type()
                return (var, ("type", expr))
            else:
                expr = self.parse_template_expr()
                return (var, ("val", expr))
        raise parse_error("instantiation")
    def parse_kernel_sig(self):
        """
        type fun(type,type,..)
        """
        return_type = self.parse_type()
        class_fun = self.parse_class_fun_name()
        self.eat(["("])
        params = []
        if self.kind in self.first_type:
            params.append(self.parse_type())
            while self.kind == ",":
                self.eat([","])
                params.append(self.parse_type())
        self.eat([")"])
        instantiations = {}
        if self.kind == "[":
            self.eat(["["])
            if self.kind == "with":
                self.eat(["with"])
            if self.kind in self.first_type:
                var, val = self.parse_instantiation()
                instantiations[var] = val
                while self.kind in [";", ","]:
                    self.eat([";", ","])
                    var, val = self.parse_instantiation()
                    instantiations[var] = val
            self.eat(["]"])
        self.eat(["EOF"])
        return dict(return_type=return_type, class_fun=class_fun,
                    params=params, instantiations=instantiations)
    def parse(self, s):
        """
        parse a string
        """
        self.init(s)
        return self.parse_kernel_sig()

class log_parser(log_parser_base):
    """
    log parser
    """
    def __init__(self, fp):
        super().__init__(fp)
        self.samples = []
        self.phase = None
        self.n_training_samples = 0
        self.loss_acc = []
        self.kernels = []
        self.key_vals = []
    def action_open_log(self, data):
        stime = time.strptime(data["when"])
        when = time.strftime("%Y-%m-%dT%H-%M-%S", stime)
        self.key_vals.append(("start_at", when))
    def action_close_log(self, data):
        stime = time.strptime(data["when"])
        when = time.strftime("%Y-%m-%dT%H-%M-%S", stime)
        self.key_vals.append(("end_at", when))
    def action_env(self, data):
        self.key_vals.append((data["var"], data["val"]))
    def action_train_batch_start(self, data):
        """
        action on train begin
        """
        self.phase = ("train", int(data["a"]), int(data["b"]))
        self.n_training_samples += int(data["b"]) - int(data["a"])
        self.samples.append(("train", []))
    def action_test_batch_start(self, data):
        """
        action on test begin
        """
        self.phase = ("test", int(data["a"]), int(data["b"]))
        self.samples.append(("test", []))
    def action_train_loss(self, data):
        """
        action on train loss
        """
        self.loss_acc.append((self.n_training_samples, "train",
                              int(data["t"]), float(data["loss"]), None))
    def action_test_loss(self, data):
        """
        action on validate loss
        """
        loss = float(data["loss"])
        acc = float(data["c"]) / float(data["n"])
        self.loss_acc.append((self.n_training_samples, "test",
                              int(data["t"]), loss, acc))
    def xxx_action_train_accuracy(self, data):
        """
        action on train accuracy
        """
        self.loss_acc.append((self.n_training_samples, "train_accuracy",
                              int(data["t"]), float(data["accuracy"])))
    def xxx_action_test_accuracy(self, data):
        """
        action on test accuracy
        """
        self.loss_acc.append((self.n_training_samples, "test_accuracy",
                              int(data["t"]), float(data["accuracy"])))
    def action_sample(self, data):
        """
        action on sample
        """
        _, cur = self.samples[-1]
        cur.append(data)
    def action_kernel_start(self, data):
        """
        action on kernel start
        """
        kernel = self.kpsr.parse(data["kernel"])
        train_test, a, b = self.phase
        # start time, end time, kernel info, elapsed time, train/validate, sample_idx0, sample_idx1
        self.kernels.append((int(data["t"]), None, kernel, None, train_test, a, b))
    def action_kernel_end(self, data):
        """
        action on kernel end
        """
        kernel = self.kpsr.parse(data["kernel"])
        kernel_time = int(data["kernel_time"])
        ker1 = self.kernels[-1]
        t0, t1, ks, kt, train_test, a, b = ker1
        assert(t1 is None), ker1
        assert(ks == kernel), ker1
        assert(kt is None), ker1
        t1 = int(data["t"])
        kt = kernel_time
        self.kernels[-1] = (t0, t1, ks, kt, train_test, a, b)
    def get_key_vals(self):
        """
        get environment variables
        """
        jsn = []
        for key, val in self.key_vals:
            jsn.append(dict(key=key, val=val))
        return jsn
    def get_samples(self):
        """
        get samples
        """
        jsn = []
        for i, (train_test, samples) in enumerate(self.samples):
            for s in samples:
                jsn.append(dict(iter=i, train_test=train_test, **s))
        return jsn
    def get_kernel_times(self):
        """
        get kernel times
        """
        jsn = []
        for t0, t1, kernel, dt, train_test, a, b in self.kernels:
            cls, cargs, fun, fargs = self.instantiate(kernel)
            if cargs is not None:
                cargs = "<%s>" % ",".join("%s" % x for x in cargs)
            if fargs is not None:
                fargs = "<%s>" % ",".join("%s" % x for x in fargs)
            jsn.append(dict(t0=t0, t1=t1, cls=cls, cargs=cargs, fun=fun, fargs=fargs, dt=dt,
                            train_test=train_test, a=a, b=b))
        return jsn
    def get_loss_accuracy(self):
        """
        get loss accuracy
        """
        data = {}
        jsn = []
        for samples, kind, t, loss, acc in self.loss_acc:
            if kind == "train":
                data = {"samples" : samples,
                        "t" : t,
                        "train_loss" : loss,
                        "train_accuracy" : "",
                        "test_loss" : "",
                        "test_accuracy" : ""}
            elif kind == "test":
                data = {"samples" : samples,
                        "t" : t,
                        "train_loss" : "",
                        "train_accuracy" : "",
                        "test_loss" : loss,
                        "test_accuracy" : acc}
            else:
                assert(kind in ["train", "test"]), kind
            jsn.append(data)
        return jsn
    def get_all_data(self):
        return "".join(self.lines)
    def write_samples_csv(self, filename):
        """
        write samples into csv
        """
        with open(filename, "w") as wp:
            csv_wp = csv.DictWriter(wp, ["iter", "train_test", "image", "pred", "truth", "sample", "t"])
            csv_wp.writeheader()
            for i, (train_test, samples) in enumerate(self.samples):
                for s in samples:
                    csv_wp.writerow(dict(iter=i, train_test=train_test, **s))
    def eval_template_arg(self, expr, env):
        """
        evaluate template arg
        """
        if isinstance(expr, type("")):
            # variable
            _, val = env[expr]
            return val
        if isinstance(expr, type(0)):
            # number
            return expr
        if isinstance(expr, type(())):
            # (op, e0, e1)
            operator, expr0, expr1 = expr
            val0 = self.eval_template_arg(expr0, env)
            val1 = self.eval_template_arg(expr1, env)
            if operator == "+":
                val = val0 + val1
            if operator == "-":
                val = val0 - val1
            if operator == "*":
                val = val0 * val1
            if operator == "/":
                val = val0 / val1
            assert(operator in ["+", "-", "*", "/", "%"]), operator
            return val
        assert(0), expr
        return None
    def instantiate(self, kernel):
        """
        instantiate kernel
        """
        # return_type = kernel["return_type"]
        class_fun = kernel["class_fun"]
        # params = kernel["params"]
        insts = kernel["instantiations"]
        class_name = class_fun["class_name"]
        class_args = class_fun["class_args"]
        fun_name = class_fun["fun_name"]
        fun_args = class_fun["fun_args"]
        if class_args is None:
            class_arg_vals = None
        else:
            class_arg_vals = [self.eval_template_arg(arg, insts) for arg in class_args]
        if fun_args is None:
            fun_arg_vals = None
        else:
            fun_arg_vals = [self.eval_template_arg(arg, insts) for arg in fun_args]
        return (class_name, class_arg_vals, fun_name, fun_arg_vals)
    def xxx_write_kernel_times_csv(self, filename):
        """
        write kernel_times into csv
        """
        with open(filename, "w") as wp:
            csv_wp = csv.DictWriter(wp, ["t0", "t1", "cls", "cargs", "fun", "fargs", "dt"])
            csv_wp.writeheader()
            for t0, t1, kernel, dt, train_test, a, b in self.kernels:
                cls, cargs, fun, fargs = self.instantiate(kernel)
                if cargs is not None:
                    cargs = "<%s>" % ",".join("%s" % x for x in cargs)
                if fargs is not None:
                    fargs = "<%s>" % ",".join("%s" % x for x in fargs)
                csv_wp.writerow(dict(t0=t0, t1=t1, cls=cls, cargs=cargs,
                                     fun=fun, fargs=fargs, dt=dt,
                                     train_test=train_test, a=a, b=b))
    def xxx_write_loss_accuracy_csv(self, filename):
        """
        write loss_accuracy into csv
        """
        with open(filename, "w") as wp:
            csv_wp = csv.DictWriter(wp, ["samples", "t",
                                         "train_accuracy", "validate_accuracy",
                                         "train_loss", "validate_loss"])
            csv_wp.writeheader()
            data = {}
            for samples, kind, t, x in self.loss_acc:
                if data.get("samples") != samples:
                    if "samples" in data:
                        csv_wp.writerow(data)
                    data = {"samples" : samples,
                            "t" : "",
                            "train_accuracy" : "",
                            "validate_accuracy" : "",
                            "train_loss" : "",
                            "validate_loss" : ""}
                data[kind] = x
                if kind == "train_accuracy":
                    data["t"] = t

def parse_log(log):
    """
    parse a log
    """
    if log == "-":
        fp = sys.stdin
    else:
        fp = open(log)
    psr = log_parser(fp)
    psr.parse_file()
    key_vals = psr.get_key_vals()
    samples = psr.get_samples()
    loss_accuracy = psr.get_loss_accuracy()
    kernel_times = psr.get_kernel_times()
    all_data = psr.get_all_data()
    classes = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]
    meta = [{"class": x} for x in classes]
    if log != "-":
        fp.close()
    return ({"key_vals"      : key_vals,
             "samples"       : samples,
             "loss_accuracy" : loss_accuracy,
             "kernel_times"  : kernel_times,
             "meta"          : meta
             },
            all_data)

def main():
    """
    main
    """
    log = sys.argv[1] if len(sys.argv) > 1 else "../vgg.log"
    plog, _ = parse_log(log)
    with open("vars.js", "w") as wp:
        wp.write("var meta_json = %s;\n"
                 % json.dumps(plog["meta"]))
        wp.write("var key_vals_json = %s;\n"
                 % json.dumps(plog["key_vals"]))
        wp.write("var samples_json = %s;\n"
                 % json.dumps(plog["samples"]))
        wp.write("var loss_accuracy_json = %s;\n"
                 % json.dumps(plog["loss_accuracy"]))
        wp.write("var kernel_times_json = %s;\n"
                 % json.dumps(plog["kernel_times"]))

def mainx():
    """
    main for test
    """
    s = ("array4<maxB, OC, H, W>&"
         " Convolution2D<maxB, IC, H, W, K, OC>::forward(array4<maxB, IC, H, W>&)"
         " [with int maxB = 64; int IC = 3; int H = 32;"
         " int W = 32; int K = 1; int OC = 16]")
    kpsr = kernel_parser()
    return kpsr.parse(s)

#main()
