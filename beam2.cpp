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
    llama_seq_id seq;
    common_sampler *smpl;
    std::vector<llama_token> tokens;
    float logit;
};
struct BeamSort{
    inline bool operator() (const Beam& a, const Beam& b){
        return a.logit > b.logit;
    }
};
struct BeamSearch{
    llama_context *ctx;
    const llama_vocab *vocab;
    int scale;
    int beam;
    llama_pos len;
    llama_batch batch;
    llama_pos pos;
    llama_seq_id seq = 1;
    std::vector<Beam> beams;
    BeamSearch(llama_context *ctx, const llama_vocab *vocab, common_sampler *smpl, std::string prompts, int scale=2, int beam=4, llama_pos len=100): ctx(ctx), vocab(vocab), batch(batch), scale(scale), beam(beam), len(len){
        batch = llama_batch_init(65536, 0, beam);
        auto prompt = common_tokenize(ctx, prompts, true);
        for(pos=0; pos<prompt.size(); pos++){
            common_batch_add(batch, prompt[pos], pos, {0}, pos==prompt.size()-1);
        }
        beams.push_back({0, smpl});
    }
    ~BeamSearch(){
        freeRight(0);
    }
    void free(int i){
        bool free = true;
        for(int j=0; j<beams.size(); j++)
            if(i != j and beams[i].seq == beams[j].seq){
                free = false;
                break;
            }
        if(free){
            llama_kv_self_seq_rm(ctx, beams[i].seq, -1, -1);
            common_sampler_free(beams[i].smpl);
        }
    }
    void freeRight(int i){
        for(; i<beams.size(); i++){
            bool free = true;
            for(int j=0; j<i; j++)
                if(beams[i].seq == beams[j].seq){
                    free = false;
                    break;
                }
            if(free){
                llama_kv_self_seq_rm(ctx, beams[i].seq, -1, -1);
                common_sampler_free(beams[i].smpl);
            }
        }
    }
    void allocRight(int i){
        for(; i<beams.size(); i++)
            for(int j=0; j<i; j++)
                if(beams[i].seq == beams[j].seq){
                    llama_kv_self_seq_cp(ctx, beams[i].seq, seq, -1, -1);
                    beams[i].seq = seq++;
                    beams[i].smpl = common_sampler_clone(beams[i].smpl);
                    break;
                }
    }
    std::string step(){
        while(beams.size()){
            llama_decode(ctx, batch);
            int size = beams.size();
            for(int i=0; i<size; i++){
                for(int n=0; n<scale; n++){
                    auto beam = beams[i];
                    auto token = common_sampler_sample(beams[i].smpl, ctx, batch.n_tokens-size+i);
                    beam.tokens.push_back(token);
                    beam.logit += llama_get_logits_ith(ctx, batch.n_tokens-size+i)[token];
                    beams.push_back(beam);
                }
                auto &beam = beams[i];
                auto token = common_sampler_sample(beams[i].smpl, ctx, batch.n_tokens-size+i);
                beam.tokens.push_back(token);
                beam.logit += llama_get_logits_ith(ctx, batch.n_tokens-size+i)[token];
            }
            common_batch_clear(batch);
            std::sort(beams.begin(), beams.end(), BeamSort());
            freeRight(beam);
            beams.resize(std::min(beam, (int)beams.size()));
            for(int i=0; i<beams.size(); i++){
                if(llama_vocab_is_eog(vocab, beams[i].tokens[beams[i].tokens.size()-1])){
                    free(i);
                    auto tokens = beams[i].tokens;
                    beams[i] = beams[beams.size()-1];
                    if(pos > len)
                        return common_detokenize(ctx, tokens, false);
                }
            }
            allocRight(0);
            for(int i=0; i<beams.size(); i++){
                common_batch_add(batch, beams[i].tokens[beams[i].tokens.size()-1], pos, {beams[i].seq}, true);
                common_sampler_accept(beams[i].smpl, beams[i].tokens[beams[i].tokens.size()-1], true);
            }
            pos++;
        }
        return "";
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
    int n_seq = llama_n_seq_max(ctx);
    auto batch = llama_batch_init(llama_n_ctx(ctx), 0, n_seq);
    BeamSearch bs(ctx, vocab, common_sampler_init(model, params.sampling), "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", n_seq, n_seq);
    std::cout << bs.step();
}