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
    int seq;
    common_sampler *smpl;
    std::string str = "";
    llama_token token = 0;
    float logit = 0;
    int cnt = 1;
};
struct BeamSort{
    inline bool operator() (const Beam& a, const Beam& b){
        return a.logit > b.logit;
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
    llama_model *model = llama_init.model.get();
    llama_context *ctx = llama_init.context.get();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_ctx = llama_n_ctx(ctx);
    int n_seq = llama_n_seq_max(ctx);
    auto batch = llama_batch_init(n_ctx, 0, 1);
    common_batch_clear(batch);
    std::vector<Beam> beam = {{0, common_sampler_init(model, params.sampling)}};
    int seq=1;
    auto prompt = common_tokenize(ctx, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", true);
    for(int i=0; i<prompt.size(); i++)
        common_batch_add(batch, prompt[i], i, {0}, i==prompt.size()-1);
    while(beam.size()){
        if(llama_decode(ctx, batch))
            return 1;
        int lim = beam.size();
        for(int i=0; i<lim; i++)
            for(int n=0; n<4; n++){
                auto token = common_sampler_sample(beam[i].smpl, ctx, batch.n_tokens-lim+i);
                beam.push_back({beam[i].seq, beam[i].smpl, beam[i].str, token, beam[i].logit+llama_get_logits_ith(ctx, batch.n_tokens-lim+i)[token], beam[i].cnt+1});
            }
        common_batch_clear(batch);
        std::sort(beam.begin(), beam.end(), BeamSort());
        for(int i=n_seq; i<beam.size(); i++){
            bool free = true;
            for(int j=0; j<i; j++)
                if(beam[i].seq==beam[j].seq){
                    free = false;
                    break;
                }
            if(free){
                llama_kv_self_seq_rm(ctx, beam[i].seq, -1, -1);
                common_sampler_free(beam[i].smpl);
            }
        }
        beam.resize(std::min(n_seq, (int)beam.size()));
        for(int i=0; i<beam.size(); i++){
            if (llama_vocab_is_eog(vocab, beam[i].token) || beam[i].cnt>100){
                std::cout << beam[i].str << "\n\n\n";
                beam[i] = beam[beam.size()-1];
                beam.pop_back();
                n_seq--;
            }
        }
        for(int i=0; i<beam.size(); i++)
            for(int j=0; j<i; j++)
                if(beam[i].seq==beam[j].seq){
                    llama_kv_self_seq_cp(ctx, beam[i].seq, seq, -1, -1);
                    beam[i].seq = seq;
                    seq++;
                    beam[i].smpl = common_sampler_clone(beam[i].smpl);
                    break;
                }
        for(int i=0; i<beam.size(); i++){
            common_batch_add(batch, beam[i].token, prompt.size()+beam[i].cnt, {beam[i].seq}, true);
            common_sampler_accept(beam[i].smpl, beam[i].token, true);
            beam[i].str += common_token_to_piece(vocab, beam[i].token);
        }
    }
}