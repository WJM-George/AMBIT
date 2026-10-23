# Third-party notices

AMBIT is developed on top of Stability AI's
[stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools).
The original MIT copyright and permission notice are retained in `LICENSE`.
The runtime has been narrowed to the FOA codec, conditioned transformer, text
conditioning, ScenePlan runtime, and inference decoding used by AMBIT.

The attribution files inherited from stable-audio-tools are retained in
`LICENSES/`, including notices for audio-diffusion-pytorch, aeiou, auraloss,
CLAP, Descript, Meta, NVIDIA, OpenCLIP, and x-transformers. Their presence does
not imply that every upstream component is distributed or required by this
inference package. Runtime dependencies retain their respective licenses.

No AMBIT or third-party model weights, tokenizer assets, or training datasets
are distributed with this source tree.

The website's headphone previews use the MIT KEMAR measurements of Gardner and
Martin, converted to SOFA by ARI. The measurement license allows unrestricted
use and requests citation of Gardner and Martin, *HRTF Measurements of a KEMAR
Dummy-Head Microphone*, MIT Media Lab Perceptual Computing Technical Report 280
(1994), and *HRTF Measurements of a KEMAR*, JASA 97(6), 3907–3908 (1995).
