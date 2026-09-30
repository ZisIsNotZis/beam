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
struct BeamTok{
    int sup;
    int tok;
};
struct Beam{
    common_sampler *smpl;
    float logit;
};
struct BeamCand{
    int sup;
    int tok;
    float logit;
};
struct BeamSearch{
    int idx;
    int pos;
    std::vector<BeamTok> toks;
    std::vector<Beam> beams;
    inline BeamSearch(llama_batch &batch, common_sampler *smpl, std::vector<int> prompt, int seq=0){
        for(pos=0; pos<prompt.size(); pos++)
            common_batch_add(batch, prompt[pos], pos, {seq}, false);
        idx = batch.n_tokens - 1;
        batch.logits[idx] = true;
        toks.push_back({});
        beams.push_back({smpl});
    }
    inline std::vector<int> step(llama_context *ctx, const llama_vocab *vocab, llama_batch &batch, int seq=0, int scale=2, int nbeam=256){
        std::vector<int> out;
        std::vector<BeamCand> cands;
        int base = toks.size();
        int sbase = base - beams.size();
        for(int i=0; i<beams.size(); i++){
            int n = 0;
            for(int j=0; j<scale; j++){
                int tok = common_sampler_sample(beams[i].smpl, ctx, idx+i);
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
                out.push_back(sbase + cands[i].sup);
                cands[i] = cands[cands.size()-1];
                cands.pop_back();
                i--;
            }
        std::vector<int> adds(cands.size());
        for(int i=0; i<cands.size(); i++)
            adds[cands[i].sup] = -1;
        idx = batch.n_tokens;
        toks.resize(base + cands.size());
        beams.resize(std::max(cands.size(), beams.size()));
        for(int i=0; i<cands.size(); i++){
            int j;
            if(adds[cands[i].sup] == -1)
                j = cands[i].sup;
            else{
                for(j=0; adds[j]; j++);
                llama_kv_self_seq_rm(ctx, seq+j, -1, -1);
                llama_kv_self_seq_cp(ctx, seq+cands[i].sup, seq+j, -1, -1);
                common_sampler_free(beams[j].smpl);
                beams[j].smpl = common_sampler_clone(beams[cands[i].sup].smpl);
            }
            common_sampler_accept(beams[j].smpl, cands[i].tok, true);
            toks[base+j].sup = sbase + cands[i].sup;
            toks[base+j].tok = cands[i].tok;
            beams[j].logit = cands[i].logit;
            adds[j] = cands[i].tok;
        }
        for(int i=0; i<adds.size(); i++)
            common_batch_add(batch, adds[i], pos, {seq+i}, true);
        for(int i=cands.size(); i<base-sbase; i++){
            llama_kv_self_seq_rm(ctx, seq+i, -1, -1);
            common_sampler_free(beams[i].smpl);
        }
        beams.resize(cands.size());
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
    inline void log(std::vector<BeamCand> cands){
        LOG_INF("beam");
        for(int i=0; i<toks.size(); i++)
            LOG_INF(" %d,%d", toks[i].sup, toks[i].tok);
        LOG_INF("\nlogit");
        for(int i=0; i<beams.size(); i++)
            LOG_INF(" %f", beams[i].logit);
        LOG_INF("\ncand");
        for(int i=0; i<cands.size(); i++)
            LOG_INF(" %d,%d,%f", cands[i].sup, cands[i].tok, cands[i].logit);
        LOG_INF("\n");
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
    BeamSearch bs(batch, common_sampler_init(model, params.sampling), common_tokenize(ctx, "<|im_start|>user\n"+params.prompt+"<|im_end|>\n<|im_start|>assistant\n", true), 0);
    while(true){
        if(llama_decode(ctx, batch))
            return 1;
        common_batch_clear(batch);
        std::vector<int> out;
        for(auto i: out = bs.step(ctx, vocab, batch, 0, 2, llama_n_seq_max(ctx)))
            std::cout << "===\n" << common_detokenize(ctx, bs.get(i), false) << "\n===\n";
        if(!out.size())
            std::cout << common_detokenize(ctx, bs.get(bs.toks.size()-1), false) << '\n';
    }
}
// export CPATH=include:ggml/include:common LIBRARY_PATH=icblas/common:icblas/bin LD_LIBRARY_PATH=icblas/bin
// g++ beam.cpp -lcommon -lllama -lggml -lggml-cpu -lggml-blas -lggml-base -lsvml -lirng -limf -lintlc -liomp5 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Ofast -s -march=native -flto=auto -obeam
// ./beam -m ~/hub/qwen2.5-0.5b-instruct-q4_k_m.gguf -np 4