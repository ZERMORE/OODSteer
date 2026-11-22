import sys
import types
import unittest
from unittest.mock import patch
import torch
from oodsteer.answers import extract, stop_reason, make_stopper

class AnswerTests(unittest.TestCase):
    def test_boxed_and_boundaries(self):
        self.assertEqual(extract(r'Thus \boxed{1,200.00}', 'gsm8k'),'1.2E+3')
        self.assertEqual(extract('So the answer is (B). Q: New? So the answer is (A).','csqa'),'B')
        self.assertEqual(extract('The answer is 20.','gsm8k'),'2E+1')
        self.assertEqual(extract('The answer is -1.2e3.','gsm8k'),'-1.2E+3')
        self.assertIsNone(extract('The answer is 2/3','gsm8k'))
        self.assertIsNone(extract('We computed 20, then 30.','gsm8k'))
        self.assertIsNone(extract('The answer must be a place.','csqa'))
        self.assertIsNone(extract(r'\boxed{\frac{1}{2}}','gsm8k'))

    def test_stop_does_not_cut_number(self):
        for text in ['The answer is 12','The answer is 120.','The answer is 120.5','The answer is 1e','The answer is 1e+']:
            self.assertIsNone(stop_reason(text,'gsm8k'),text)
        self.assertEqual(stop_reason('The answer is 120.5. ','gsm8k'),'complete-answer')
        self.assertEqual(stop_reason(r'Thus \boxed{120.5}','gsm8k'),'complete-answer')
        self.assertEqual(stop_reason('The answer is (B)','csqa'),'complete-answer')
        self.assertEqual(stop_reason('The answer is (B) Q:','csqa'),'next-question')
        self.assertIsNone(stop_reason('The answer is (B)','csqa','question'))
        self.assertIsNone(stop_reason('Q: New?','csqa','eos'))

    def test_first_answer_protocol(self):
        self.assertEqual(extract('The answer is 20. The answer is 30.','gsm8k'),'2E+1')
        self.assertEqual(extract('So, the answer is (d).','csqa'),'D')
        self.assertEqual(extract('The correct answer is (e).','csqa'),'E')
        self.assertEqual(extract('So the answer is yes.','strategyqa'),'yes')

    def test_stopper_ignores_prompt(self):
        class Tokenizer:
            def decode(self,ids,**kw): return ''.join(chr(i) for i in ids)
        module=types.SimpleNamespace(StoppingCriteria=object,StoppingCriteriaList=list)
        with patch.dict(sys.modules,transformers=module):
            prompt='The answer is 99. Q: test A:'
            stopper,criteria=make_stopper(Tokenizer(),len(prompt),'gsm8k','answer')
            def step(text):
                return stopper(torch.tensor([[ord(c) for c in prompt+text]]),None)
            self.assertFalse(step('The answer is 12'))
            self.assertFalse(step('The answer is 120.5'))
            self.assertTrue(step('The answer is 120.5. '))
            self.assertEqual(stopper.reason,'complete-answer')

    def test_real_hf_generate(self):
        try:
            from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessor
        except ImportError:
            self.skipTest("transformers not installed in this local interpreter")
        script="The answer is 120.5. EXTRA"
        class Forced(LogitsProcessor):
            def __call__(self,ids,scores):
                scores.fill_(float("-inf"))
                scores[:,ord(script[ids.shape[1]-1])]=0
                return scores
        class Tokenizer:
            def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
        config=LlamaConfig(vocab_size=128,hidden_size=16,intermediate_size=32,
                           num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                           bos_token_id=1,eos_token_id=2,pad_token_id=0)
        model=LlamaForCausalLM(config).eval()
        stopper,criteria=make_stopper(Tokenizer(),1,'gsm8k','answer')
        with torch.no_grad():
            output=model.generate(torch.tensor([[1]]),max_new_tokens=len(script),
                logits_processor=[Forced()],stopping_criteria=criteria,do_sample=False)
        result=Tokenizer().decode(output[0,1:])
        self.assertEqual(result,'The answer is 120.5. ')
        self.assertEqual(extract(result,'gsm8k'),'120.5')
        self.assertEqual(stopper.reason,'complete-answer')
