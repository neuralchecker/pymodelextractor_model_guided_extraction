from collections.abc import Iterable
from functools import reduce
from typing import List, Set, FrozenSet, Any, NoReturn
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.abstract.boolean_model import BooleanModel
from inspect import signature


class ComposedBooleanModel(BooleanModel):

    def __init__(self, models : list, alphabet : Alphabet, reduction_function = lambda *args:all(args), verbose:bool = False, verify_red_fun:bool = True):
        '''__init__ will call reduction_function when initiating the object in order to verify it will be able 
        to call it when calling accept, unless verify_red_fun is set to False'''
        #verify parameters
        if verify_red_fun:
            self._verify_reduction_function(reduction_function, models)
        self._verify_models(models)

        #init object
        self._models =  models
        self._alphabet = alphabet
        self._verbose = verbose
        self._reduction_function = reduction_function
        
    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    @property
    def name(self) -> str:
        name = "Composed_model:"
        for model in self._models:
            name = name +" - "+ model._name
        return name
        
    def accepts(self, sequence: Sequence) -> bool:
        accepts = list(map(lambda x: x.accepts(sequence), self._models))
        result = self._reduction_function(*accepts)
        if self._verbose and result:
            print(self.name + " accepted: ", sequence)
        return result

    def accepts_batch(self, sequences):#-> list[bool]:
        for model in self._models:
            assert(hasattr(model, 'accepts_batch'))
        results = list(map(lambda x: x.accepts_batch(sequences), self._models))
        intermediate_results = zip(*results)
        return list(map(lambda x:self._reduction_function(*x),intermediate_results))

    def _verify_reduction_function(self, fun: "Func", models: list) -> NoReturn:
        try:
            args = map(lambda x: True, models)
            fun(*args)
        except:
            print("Reduction function should accept as many parameters as len(models)")
            raise

    def _verify_models(self, models: Any) -> NoReturn:
        all_verify = all(map(self._check_models_accepts_sequence, models))
        if not all_verify:
            print("Make sure all the models have a method accepts(s: Sequence)")
        assert(all_verify)


    def _check_models_accepts_sequence(self, m: Any) -> bool:
        try: 
            params=signature(m.accepts).parameters
            return len(params) > 0
        except:
            return False

