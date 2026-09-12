import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_decision_retention import selected_margin_penalty


def test_original_native_distribution_gets_no_artificial_sharpening_gradient():
    original=torch.tensor([.0071,0.,-.01],requires_grad=True)
    loss,details=selected_margin_penalty(original,original,0)
    loss.backward()
    assert float(loss)==0 and torch.equal(original.grad,torch.zeros_like(original))
    assert details['selected_preserved']


def test_nearly_flat_large_vocabulary_can_flip_despite_tiny_kl():
    teacher=torch.zeros(648);teacher[0]=.0071
    student=teacher.clone();student[1]=.01;student.requires_grad_()
    p,q=student.double().log_softmax(0),teacher.double().log_softmax(0)
    kl=(q.exp()*(q-p)).sum()
    penalty,details=selected_margin_penalty(student,teacher,0)
    assert float(kl)<1e-6 and not details['selected_preserved']
    assert float(penalty)>0
    penalty.backward()
    assert student.grad[0]<0 and student.grad[1]>0


def test_new_competitor_is_checked_over_the_complete_legal_support():
    teacher=torch.tensor([1.,.8,-3.],requires_grad=True)
    student=torch.tensor([1.,.7,1.1],requires_grad=True)
    loss,details=selected_margin_penalty(student,teacher,0)
    loss.backward()
    assert teacher.grad is None
    assert not details['selected_preserved'] and student.grad[2]>0 and student.grad[1]==0


def test_self_margin_does_not_prescribe_an_unselected_action():
    with pytest.raises(ValueError,match='original native argmax'):
        selected_margin_penalty(torch.tensor([1.,2.]),torch.tensor([1.,2.]),0)


@pytest.mark.parametrize('fraction',[0.,2.,float('nan')])
def test_invalid_margin_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        selected_margin_penalty(torch.tensor([1.,0.]),torch.tensor([1.,0.]),0,keep_fraction=fraction)
