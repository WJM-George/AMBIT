"""Publish separate, auditable results for the three held-out test cohorts."""
from common import *
import csv


def main():
    selection=read(ROOT/'SELECTION.json');step=selection['selected_checkpoint_steps']
    assert selection['selection_split']=='validation' and not selection['test_used_for_selection']
    keys=['audio_codec_foa_nmse','audio_w_mrstft','independent_clap_output_target_cosine',
        'doa_target_mean_deg','unchanged_demix_si_sdr_delta_vs_copy_db','activity_output_target_iou']
    lines=['# Editing DiT 175 万训练对、续训 30k','',
        f'从原始 50k warm start。六个续训检查点累计 55k–80k；同一批固定 2,000 条 validation 选出的模型：{step:,}（本轮 +{step-50000:,}）。',
        '三套 test 分别报告，不把追加集结果与原 5k 的历史均值混为一行。模型选择只读取 validation。','',
        '| 测试集 | 样本数 | codec FOA NMSE ↓ | W MRSTFT ↓ | CLAP target ↑ | DOA ° ↓ | 保留声源 ΔSI-SDR ↑ | 活动 IoU ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    reviews={};records=[]
    for scope,expected in [('test_original',5000),('test_addition250k',1250),('test_spatial_multi500k',2500)]:
        directory=ROOT/scope/f'STEP{step:06d}';review=read(directory/'RESULT.json');summary=read(directory/'SUMMARY.json')
        assert review['status']=='PASS_FULL_SPLIT_INDEPENDENT_CPU_REVIEW' and review['rows']==expected
        assert review['checkpoint_steps']==step and sha(directory/'SUMMARY.json')==review['summary_sha256']
        overall=summary['all_finite/all'][str(step)];values=[overall[k]['mean'] for k in keys]
        lines.append('| '+scope+' | '+str(expected)+' | '+' | '.join('—' if v is None else f'{v:.5f}' for v in values)+' |')
        reviews[scope]=dict(path=str(directory/'RESULT.json'),sha256=sha(directory/'RESULT.json'),rows=expected)
        for group,model in summary.items():
            for metric,value in model[str(step)].items():records.append([scope,expected,step,group,metric,value['mean'],value['rows'],value['group_rows'],value['coverage_fraction']])
    lines+=['','上表对整套测试集的有效值求均值；CLAP target 覆盖所有可评分编辑。SUMMARY 同时保留旧评分器按增删事件限定 CLAP 的历史口径，新增 all_finite/ 前缀用于全类报告。逐指标有效样本数、分事件类型、声源数量、内容类别和长度桶见 `THREE_TEST_METRICS.csv`；不可用项保留为空，未填零。',
        '多声源空间/波形指标沿用相同原生 FOA 评分公式。没有可核验的未编辑声源时，保留指标不适用。新集原始 GT 按冻结配方重渲染并要求 PCM24 FLAC 哈希一致。',
        '所有测试输出保留完整 FOA 音频，可用于后续同一候选的试听和论文指标复核。训练集只使用已有 FOA VAE latent 与完整 NEW ScenePlan。',
        f'\n选模证据：{ROOT/"SELECTION.json"}\n模型指针：{ROOT/"BEST_CHECKPOINT.json"}\n']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    with (ROOT/'THREE_TEST_METRICS.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['testset','test_rows','checkpoint','group','metric','mean','valid_rows','eligible_rows','coverage_fraction']);writer.writerows(records)
    result=dict(status='COMPLETE_1750K_TRAINING_30K_SIX_CHECKPOINTS_VALIDATION2K_AND_THREE_TESTS',
        training_rows=1750000,new_training_steps=30000,checkpoint_steps=STEPS[1:],selected_checkpoint_steps=step,
        validation_rows=2000,test_rows=8750,test_cohorts=reviews,selection_sha256=sha(ROOT/'SELECTION.json'),
        report=str(ROOT/'REPORT.md'),metrics=str(ROOT/'THREE_TEST_METRICS.csv'),completed_at=now(),human_rated=False)
    write(ROOT/'RESULT.json',result);REPORT.mkdir(exist_ok=True,parents=True);write(REPORT/'RESULT.json',result)
    (REPORT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
