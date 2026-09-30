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
struct Beam{
    int seq;
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
    inline BeamSearch(llama_batch &batch, std::vector<int> prompt, int &seq){
        for(pos=0; pos<prompt.size(); pos++)
            common_batch_add(batch, prompt[pos], pos, {seq}, false);
        idx = batch.n_tokens - 1;
        batch.logits[idx] = true;
        toks.push_back({});
        beams.push_back({seq++});
    }
    inline int step(llama_context *ctx, const llama_vocab *vocab, llama_batch &batch, common_sampler *smpl, int &seq, int scale, int nbeam){
        int out = 0;
        std::vector<BeamCand> cands;
        int toksize = toks.size();
        int beamsize = beams.size();
        for(int i=0; i<beamsize; i++){
            int n = 0;
            auto logit = llama_get_logits_ith(ctx, idx+i);
            for(int j=0; j<scale; j++){
                int tok = common_sampler_sample(smpl, ctx, idx+i);
                if(std::none_of(cands.end()-n, cands.end(), [&](const BeamCand &cand){return cand.tok==tok;})){
                    cands.push_back({i, tok, beams[i].logit+logit[tok]});
                    n++;
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [&](const BeamCand &a, const BeamCand &b){return a.logit>b.logit;});
        cands.resize(std::min(nbeam, (int)cands.size()));
        for(int i=0; i<cands.size(); i++)
            if(llama_vocab_is_eog(vocab, cands[i].tok)){
                out = toksize - beamsize + cands[i].sup;
                cands[i] = cands[cands.size()-1];
                cands.pop_back();
                break;
            }
        int candsize = cands.size();
        std::vector<bool> use(beamsize);
        std::vector<Beam> newbeams;
        idx = batch.n_tokens;
        for(int i=0; i<candsize; i++){
            int n;
            if(use[cands[i].sup]){
                n = seq++;
                llama_kv_self_seq_cp(ctx, beams[cands[i].sup].seq, n, -1, -1);
            }else{
                n = beams[cands[i].sup].seq;
                use[cands[i].sup] = true;
            }
            common_batch_add(batch, cands[i].tok, pos, {n}, true);
            toks.push_back({toksize-beamsize+cands[i].sup, cands[i].tok});
            newbeams.push_back({n, cands[i].logit});
        }
        for(int i=0; i<beamsize; i++)
            if(!use[i])
                llama_kv_self_seq_rm(ctx, beams[i].seq, -1, -1);
        beams = std::move(newbeams);
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
    int seq = 0;
    BeamSearch bs(batch, common_tokenize(ctx, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+params.prompt+"<|im_end|>\n<|im_start|>assistant\n", true), seq);
    while(true){
        long t = ggml_time_ms();
        if(llama_decode(ctx, batch))
            return 1;
        common_batch_clear(batch);
        long T = ggml_time_ms();
        int out = bs.step(ctx, vocab, batch, smpl, seq, 2, llama_n_seq_max(ctx));
        LOG_INF("kv%d pos%d beam%ld t%ld/%ld\n", llama_kv_self_used_cells(ctx), bs.pos, bs.beams.size(), T-t, ggml_time_ms()-t);
        if(out)
            LOG_INF("%s\n", common_detokenize(ctx, bs.get(out), false).c_str());
    }
}
/*
export CPATH=include:ggml/include:common LIBRARY_PATH=icblas/common:icblas/bin LD_LIBRARY_PATH=icblas/bin
g++ beam.cpp -lcommon -lllama -lggml -lggml-cpu -lggml-blas -lggml-base -lsvml -lirng -limf -lintlc -liomp5 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Ofast -s -march=native -flto=auto -obeam
./beam -m ~/hub/qwen2.5-0.5b-instruct-q4_k_m.gguf --temp .7 --top-p .8 --repeat-penalty 1.4 --repeat-last-n 128 -p 'tell me a joke' -np 2

export CPATH=../include:../ggml/include:../common LIBRARY_PATH=common:bin LD_LIBRARY_PATH=bin
g++ ~/beam/beam7.cpp -lcommon -lllama -lggml -lggml-cpu -lggml-base -lggml-cuda -Ofast -s -march=native -flto=auto -obeam

./beam -m ~/qwen0.qwendat.qwen.iq3xxs.gguf -t 6 --prio 2 --prio-batch 2 -ngl 99 -dev cuda0 -sm none -mg 0 -fa --temp .7 --top-p .8 --repeat-penalty 1.4 --repeat-last-n 128 -p '写一篇1000字的关于母爱的散文。' -np 8
*/