"""Original scoring resolver with hash-verified parity stems for new additions."""
import copy
from common import *
from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_audio_end_to_end import OfflineTruthResolver

class ExpandedTruthResolver(OfflineTruthResolver):
    def row(self,ordinal):
        self.current_ordinal=ordinal
        return super().row(ordinal)

    def _manifest_row(self,manifest_path,expected_sha,sample_id):
        original=super()._manifest_row(manifest_path,expected_sha,sample_id)
        source_result=json.loads(original['render_result_json'])
        if source_result.get('stem_refs'):return original
        import pyarrow.parquet as pq
        row=self.connection.execute('SELECT * FROM pairs WHERE pair_ordinal=?',(self.current_ordinal,)).fetchone()
        assert row['pair_id'].startswith('speditadd250kv1_')
        manifest=row['target_materialized_manifest_path'];assert sha(manifest)==row['target_materialized_manifest_sha256']
        matches=[r for r in pq.read_table(manifest).to_pylist() if r['pair_id']==row['pair_id']];assert len(matches)==1
        target=json.loads(matches[0]['target_render_result_json'])
        assert digest(target)==row['target_render_result_sha256']
        assert target['source_parity_verified'] and target['source_parity_sha256']==row['source_foa_sha256']
        assert target['source_sample_id']==sample_id and target['source_parity_stem_refs']
        source_result['stem_refs']=target['source_parity_stem_refs']
        # This in-memory view supplies independently rendered source stems for
        # offline scoring; neither stored source metadata nor model inputs change.
        return dict(original,render_result_json=canonical(source_result))
