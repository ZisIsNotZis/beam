#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iostream>
struct Beam{
    int sup;
    int token;
};
struct BeamCand{
    int sup;
    int token;
    float logit;
    int i;
};
struct BeamSearch{
    int idx;
    int pos;
    std::vector<Beam> beams;
    std::vector<float> logits;
    inline BeamSearch(llama_batch &batch, std::vector<int> prompt, int seq=0){
        for(pos=0; pos<prompt.size(); pos++)
            common_batch_add(batch, prompt[pos], pos, {seq}, false);
        idx = batch.n_tokens - 1;
        batch.logits[idx] = true;
        beams.push_back({});
        logits.push_back(0);
    }
    inline std::vector<int> step(llama_context *ctx, const llama_vocab *vocab, common_sampler *smpl, llama_batch &batch, int seq=0, int scale=2, int nbeam=256){
        if(!beams.size())
            return {};
        std::vector<int> out;
        std::vector<BeamCand> cands;
        int base = beams.size();
        int sbase = base - logits.size();
        for(int i=0; i<logits.size(); i++){
            int n = 0;
            for(int j=0; j<scale; j++){
                int token = common_sampler_sample(smpl, ctx, idx+i);
                if(std::none_of(cands.end()-n, cands.end(), [&](BeamCand cand){return cand.token==token;})){
                    cands.push_back({i, token, logits[i]+llama_get_logits_ith(ctx, idx+i)[token]});
                    n++;
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [&](BeamCand a, BeamCand b){return a.logit>b.logit;});
        cands.resize(std::min(nbeam, (int)cands.size()));
        for(int i=0; i<cands.size(); i++)
            if(llama_vocab_is_eog(vocab, cands[i].token)){
                out.push_back(sbase+cands[i].sup);
                cands[i] = cands[cands.size()-1];
                cands.pop_back();
                i--;
            }
        std::vector<char> is(cands.size());
        for(int i=0; i<cands.size(); i++)
            is[cands[i].sup] = 1;
        for(int i=0; i<cands.size(); i++){
            int j;
            if(is[cands[i].sup] == 1)
                j = cands[i].sup;
            else{
                for(j=0; is[j]; j++);
                // LOG_INF("cp %d %d\n", seq+cands[i].sup, seq+j);
                llama_kv_self_seq_rm(ctx, seq+j, -1, -1);
                llama_kv_self_seq_cp(ctx, seq+cands[i].sup, seq+j, -1, -1);
            }
            cands[i].i = j;
            is[j] = 2;
        }
        std::sort(cands.begin(), cands.end(), [](BeamCand a, BeamCand b){return a.i<b.i;});
        idx = batch.n_tokens;
        beams.resize(base+cands.size());
        logits.resize(cands.size());
        for(int i=0; i<cands.size(); i++){
            // LOG_INF("add %d %d %d\n", seq+i, pos, cands[i].token);
            common_batch_add(batch, cands[i].token, pos, {seq+i}, true);
            common_sampler_accept(smpl, cands[i].token, true);
            beams[base+i].sup = sbase + cands[i].sup;
            beams[base+i].token = cands[i].token;
            logits[i] = cands[i].logit;
        }
        for(int i=cands.size(); i<base-sbase; i++){
            // LOG_INF("rm %d\n", seq+i);
            llama_kv_self_seq_rm(ctx, seq+i, -1, -1);
        }
        pos++;
        return out;
    }
    inline std::vector<int> get(int i){
        std::vector<int> tokens;
        while(i){
            tokens.push_back(beams[i].token);
            i = beams[i].sup;
        }
        std::reverse(tokens.begin(), tokens.end());
        return tokens;
    }
    inline void free(llama_context *ctx, int seq=0){
        for(int i=0; i<logits.size(); i++)
            llama_kv_self_seq_rm(ctx, seq+i, -1, -1);
    }
    inline void log(std::vector<BeamCand> cands){
        LOG_INF("beam");
        for(int i=0; i<beams.size(); i++)
            LOG_INF(" %d,%d", beams[i].sup, beams[i].token);
        LOG_INF("\nlogit");
        for(int i=0; i<logits.size(); i++)
            LOG_INF(" %f", logits[i]);
        LOG_INF("\ncand");
        for(int i=0; i<cands.size(); i++)
            LOG_INF(" %d,%d,%f", cands[i].sup, cands[i].token, cands[i].logit);
        LOG_INF("\n");
    }

};
int main(int argc, char ** argv){
    common_params params;
    common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL);
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result llama_init = common_init_from_params(params);
    llama_context *ctx = llama_init.context.get();
    llama_model *model = llama_init.model.get();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    auto smpl = common_sampler_init(model, params.sampling);
    auto batch = llama_batch_init(llama_n_ctx(ctx), 0, llama_n_seq_max(ctx));
    auto prompt = common_tokenize(ctx, "<|im_start|>user\nWrite a 1000 word story about Cinderella fighting monsters. Be creative.<|im_end|>\n<|im_start|>assistant\n", true);
    BeamSearch bs(batch, prompt, 0);
    while(true){
        llama_decode(ctx, batch);
        common_batch_clear(batch);
        bs.step(ctx, vocab, smpl, batch, 0, 2, llama_n_seq_max(ctx));
        LOG_INF("%s\n===\n", common_detokenize(ctx, bs.get(bs.beams.size()-1), true).c_str());
    }
}
/*
cd ~/lc.git/bin
export CPATH=include:ggml/include:common LIBRARY_PATH=gcblas/common:gcblas/bin LD_LIBRARY_PATH=gcblas/bin
g++ beam4.cpp -lllama -lggml -lcommon -lggml-base -Ofast -s -march=native -flto=auto
./a.out -m ~/hub/qwen2.5-0.5b-instruct-q4_k_m.gguf -np 2
*/