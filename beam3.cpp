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
    std::vector<int> tokens;
    float logit;
};
struct BeamCand{
    int i;
    int token;
    float logit;
};
bool unqLeft(std::vector<BeamCand> cands, int i){
    for(int j=0; j<i; i++)
        if(cands[i].i == cands[j].i)
            return false;
    return true;
};
bool unq(std::vector<BeamCand> cands, int i){
    for(int j=0; j<cands.size(); i++)
        if(i != j && cands[i].i == cands[j].i)
            return false;
    return true;
};
struct BeamSearch{
    int idx;
    int pos;
    std::vector<Beam> beams;
    BeamSearch(llama_batch &batch, std::vector<int> prompt, int seq=0){
        idx = batch.n_tokens;
        for(pos=0; pos<prompt.size(); pos++){
            common_batch_add(batch, prompt[pos], pos, {seq}, false);
        }
        batch.logits[idx+pos-1] = true;
        beams.push_back({});
    }
    std::vector<int> operator()(llama_context *ctx, const llama_vocab *vocab, common_sampler *smpl, llama_batch &batch, int seq=0, int nbeam=4, int len=50){
        int size = beams.size();
        std::vector<BeamCand> cands;
        for(int i=0; i<size; i++)
            for(int j=0; j<nbeam; j++){
                auto token = common_sampler_sample(smpl, ctx, idx-size+i);
                cands.push_back({i, token, beams[i].logit+llama_get_logits_ith(ctx, idx-size+i)[token]});
            }
        std::sort(cands.begin(), cands.end(), [](const BeamCand &a, const BeamCand &b){return a.logit>b.logit;});
        for(int i=nbeam; i<cands.size(); i++)
            if(unqLeft(cands, i))
                llama_kv_self_seq_rm(ctx, seq+cands[i].i, -1, -1);
        cands.resize(std::min(nbeam, (int)cands.size()));
        for(int i=0; i<cands.size(); i++)
            if(llama_vocab_is_eog(vocab, idx-size+cands[i].i)){
                if(unq(cands, i))
                    llama_kv_self_seq_rm(ctx, seq+cands[i].i, -1, -1);
                int j = cands[i].i;
                cands[i] = cands[cands.size()-1];
                cands.pop_back();
                i--;
                if(pos >= len)
                    return beams[j].tokens;
            }
        std::set<int> is;
        for(int i=0; i<cands.size(); i++)
            is.insert(cands[i].i);
        beams.resize(cands.size());
        idx = batch.n_tokens;
        for(int i=0; i<cands.size(); i++)
            if(!unqLeft(cands, i)){
                int j;
                for(j=0; is.find(j)!=is.end(); j++);
                beams[j] = beams[cands[i].i];
                beams[j].tokens.push_back(cands[i].token);
                beams[j].logit += cands[i].logit;
                common_batch_add(batch, cands[i].token, pos, {seq+j}, true);
            }
        for(int i=0; i<cands.size(); i++)
            if(unqLeft(cands, i)){
                beams[cands[i].i].tokens.push_back(cands[i].token);
                beams[cands[i].i].logit += cands[i].logit;
                common_batch_add(batch, cands[i].token, pos, {seq+cands[i].i}, true);
            }
        pos++;
        return beams[0].tokens;
    }
    void free(llama_context *ctx, int seq=0){
        for(int i=0; i<beams.size(); i++)
            llama_kv_self_seq_rm(ctx, seq+i, -1, -1);
    }
};
int main(int argc, char ** argv){
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL))
        return 1;
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result llama_init = common_init_from_params(params);
    llama_context *ctx = llama_init.context.get();
    llama_model *model = llama_init.model.get();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    auto smpl = common_sampler_init(model, params.sampling);
    auto batch = llama_batch_init(llama_n_ctx(ctx), 0, 4*llama_n_seq_max(ctx));
    auto prompt = common_tokenize(ctx, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello=<|im_end|>\n<|im_start|>assistant\n", true);
    std::vector<BeamSearch> bs;
    for(int j=0; j<4; j++)
        bs.push_back({batch, prompt, j*llama_n_seq_max(ctx)});
    for(int i=0; i<10; i++){
        if(llama_decode(ctx, batch))
            return 1;
        common_batch_clear(batch);
        for(int j=0; j<4; j++)
            std::cout << common_detokenize(ctx, bs[j](ctx, vocab, smpl, batch, j*llama_n_seq_max(ctx), llama_n_seq_max(ctx)), true) << '\n';
    }
}