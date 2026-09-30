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
#include <numeric>
struct BeamTok{
    int sup;
    int tok;
};
template<bool hasSmpl=false>struct Beam{
    float logit;
};
template<>struct Beam<true>{
    float logit;
    common_sampler *smpl;
};
struct BeamCand{
    int sup;
    int tok;
    float logit;
};
template<bool hasSmpl=false>struct BeamSearch{
    int idx;
    int pos;
    std::vector<BeamTok> toks;
    std::vector<Beam<hasSmpl>> beams;
    inline BeamSearch(llama_batch &batch, common_sampler *smpl, std::vector<int> prompt, int seq=0){
        for(pos=0; pos<prompt.size(); pos++)
            common_batch_add(batch, prompt[pos], pos, {seq}, false);
        idx = batch.n_tokens - 1;
        batch.logits[idx] = true;
        toks.push_back({});
        beams.push_back({});
        if constexpr(hasSmpl)
            beams[0].smpl = smpl;
    }
    inline std::vector<int> step(llama_context *ctx, const llama_vocab *vocab, llama_batch &batch, common_sampler *smpl, int seq=0, int scale=2, int nbeam=256){
        std::vector<int> out;
        std::vector<BeamCand> cands;
        int toksize = toks.size();
        int beamsize = beams.size();
        for(int i=0; i<beamsize; i++){
            int n = 0;
            for(int j=0; j<scale; j++){
                int tok;
                if constexpr(hasSmpl)
                    tok = common_sampler_sample(beams[i].smpl, ctx, idx+i);
                else
                    tok = common_sampler_sample(smpl, ctx, idx+i);
                if(std::none_of(cands.end()-n, cands.end(), [&](BeamCand cand){return cand.tok==tok;})){
                    cands.push_back({i, tok, beams[i].logit+llama_get_logits_ith(ctx, idx+i)[tok]});
                    n++;
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [&](BeamCand a, BeamCand b){return a.logit>b.logit;});
        cands.resize(std::min(nbeam, (int)cands.size()));
        for(int i=0; i<cands.size(); i++)
            if(llama_vocab_is_eog(vocab, cands[i].tok)){
                out.push_back(toksize - beamsize + cands[i].sup);
                cands[i] = cands[cands.size()-1];
                cands.pop_back();
                i--;
            }
        int candsize = cands.size();
        int maxsize = std::max(beamsize, candsize);
        std::vector<char> is(maxsize);
        for(int i=0; i<candsize; i++)
            is[cands[i].sup] = 1;
        idx = batch.n_tokens;
        toks.resize(toksize + candsize);
        beams.resize(maxsize);
        for(int i=0; i<candsize; i++){
            int j;
            if(is[cands[i].sup] == 1)
                j = cands[i].sup;
            else{
                for(j=0; is[j]; j++);
                if(j < beamsize){
                    llama_kv_self_seq_rm(ctx, seq+j, -1, -1);
                    if constexpr(hasSmpl)
                        common_sampler_free(beams[j].smpl);
                }
                llama_kv_self_seq_cp(ctx, seq+cands[i].sup, seq+j, -1, -1);
                if constexpr(hasSmpl)
                    beams[j].smpl = common_sampler_clone(beams[cands[i].sup].smpl);
            }
            toks[toksize+j].sup = toksize - beamsize + cands[i].sup;
            toks[toksize+j].tok = cands[i].tok;
            beams[j].logit = cands[i].logit;
            is[j] = 2;
        }
        for(int i=0; i<candsize; i++){
            common_batch_add(batch, toks[toksize+i].tok, pos, {seq+i}, true);
            if constexpr(hasSmpl)
                common_sampler_accept(beams[i].smpl, toks[toksize+i].tok, true);
        }
        for(int i=candsize; i<beamsize; i++){
            llama_kv_self_seq_rm(ctx, seq+i, -1, -1);
            if constexpr(hasSmpl)
                common_sampler_free(beams[i].smpl);
        }
        beams.resize(candsize);
        pos++;
        return out;
    }
    inline std::vector<int> get(int i){
        std::vector<int> out;
        while(i){
            out.push_back(toks[i].tok);
            i = toks[i].sup;
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
    inline void free(llama_context *ctx, int seq=0){
        for(int i=0; i<beams.size(); i++)
            llama_kv_self_seq_rm(ctx, seq+i, -1, -1);
    }
};
int main(int argc, char ** argv){
    common_params params;
    if(!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PARALLEL))
        return 1;
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result llama_init = common_init_from_params(params);
    llama_context *ctx = llama_init.context.get();
    llama_model *model = llama_init.model.get();
    const llama_vocab *vocab = llama_model_get_vocab(model);
    auto batch = llama_batch_init(llama_n_ctx(ctx), 0, llama_n_seq_max(ctx));
    auto smpl = common_sampler_init(model, params.sampling);
    BeamSearch bs(batch, smpl, common_tokenize(ctx, "<|im_start|>user\n"+params.prompt+"<|im_end|>\n<|im_start|>assistant\n", true), 0);
    while(true){
        LOG_INF("kv:%d/%d tok:%d/%d pos:%d beam:%ld time:",
            llama_kv_self_used_cells(ctx), llama_kv_self_n_tokens(ctx),
            batch.n_tokens, std::accumulate(batch.n_seq_id,batch.n_seq_id+batch.n_tokens,0),
            bs.pos,
            bs.beams.size()
        );
        auto t = ggml_time_ms();
        if(llama_decode(ctx, batch))
            return 1;
        common_batch_clear(batch);
        LOG_INF("%ld plantime:", ggml_time_ms()-t);
        t = ggml_time_ms();
        std::vector<int> out = bs.step(ctx, vocab, batch, smpl, 0, 2, llama_n_seq_max(ctx));
        LOG_INF("%ld\n", ggml_time_ms()-t);
        for(auto i: out)
            LOG_INF("===\n%s\n", common_detokenize(ctx, bs.get(i), false).c_str());
    }
}
/*
export CPATH=include:ggml/include:common LIBRARY_PATH=icblas/common:icblas/bin LD_LIBRARY_PATH=icblas/bin
g++ beam.cpp -lcommon -lllama -lggml -lggml-cpu -lggml-blas -lggml-base -lsvml -lirng -limf -lintlc -liomp5 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Ofast -s -march=native -flto=auto -obeam
./beam -m ~/hub/qwen2.5-0.5b-instruct-q4_k_m.gguf --temp .7 --top-p .8 --repeat-penalty 1.4 --repeat-last-n 128 -p 'tell me a joke' -np 2
*/