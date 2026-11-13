# engine3d cheat sheet
Import: `from common.engine3d import *`. Scene class: `class X(Stage3D)`; start with `self.setup_stage("iso")`.
Colors: black/white only; `ACCENT` (blue) ONLY to show change (firing, selection, result).
Views: iso flat top side front back34 low high. `self.view(name, run)`, `tour([..])`, `orbit()`, `dolly()`, `spin()`.
2D content inside a 3D scene: use `self.view("flat", 0.01)`; normal Tex/MathTex/ImageMobject work there.
Objects: box cylinder sphere cone torus arrow3d segment axes3d grid_floor curve3d voxels cnn_stack
Surfaces: surface bowl_surface saddle_surface gaussian_surface
Data: scatter_classes plane_poly bars3d matrix_grid grid_cell
Nets: Net3D(layers).reveal/forward_pass/fire_layer/reset/fade ; Perceptron3D(n_in).reveal/fire/reset/fade
Stage: caption label3d reveal keep clear_scene blue unblue pulse signal
Never wrap DashedLine inside DashedVMobject. Only smooth/rush_into/rush_from/linear rate functions.

# engine3d_ml cheat sheet (from lib3d import *)
Helpers: lit(m) / unlit(m) (blue on/off), tags(scene,[(text,pos)]), flashes(scene,[(a,b)])
CNN: ConvSweep(n,k).reveal/sweep/reset ; PoolDemo().reveal/pool ; CNN3D(spec).reveal/forward/reset
Sequence: RNNUnrolled3D(steps).reveal/run/reset ; LSTMCell3D().reveal/fire/reset ; Attention3D(n).reveal/focus/unfocus/sweep
Training: CompGraph3D.linear_loss().reveal/forward/backward/reset ; LossLandscape().reveal/descend(lr,steps,momentum,dashed)
          gradient_arrow(fn,p) ; Autoencoder3D().reveal/squeeze ; dropout/undrop(scene,net,layer,idx) ; backprop(scene,net)
          ResBlock3D().reveal/fire/reset
Classic: sigmoid_surface boundary_line blobs2d ; KMeans3D().reveal/step(i) ; SVMMargin3D().reveal/show_margin/highlight_support
         DecisionTree3D().reveal/fire_path/reset ; embedding_cloud + show_neighbors ; batch_stack
All reveal() calls auto-register objects, so scene.clear_scene() removes them.

# Layout
lib3d/core/engine3d.py  Stage3D + primitives + Net3D/Perceptron3D
lib3d/ml/{helpers,cnn,sequence,training,classic}.py  ML components
lib3d/tests/test_3d_ml.py  smoke test:  bash lib3d/run_tests.sh
Chapter usage:  from lib3d import *   (see scenes/_chapter_template.py)

# lib3d/ml/basics.py  (Chapter 5: ML basics)  - bash lib3d/run_tests.sh ch05
CapacityDemo3D().reveal/fit(degree)      view("front")   underfit/good/overfit
LinearFit3D().reveal/fit/residuals       view("front")
KFold3D().reveal/cycle ; KNN3D().reveal/classify(k) ; PCA3D().reveal/show_axes/project(k)/unproject
SGDLandscape().reveal/descend/descend_noisy ; MinibatchDemo().reveal/draw/sweep
MLRecipe3D().reveal/run/reset ; CurseOfDim3D().reveal/sparse ; ManifoldDemo().reveal/unroll/roll

# CAMERA RULE: "front" looks along x -> x-laid-out diagrams collapse to a line. Use "slant" (x-z plane
# with depth) or "oblique" (general 3D). "side" is flat face-on, only for deliberate 2D-style shots.

# lib3d/ml/estimation.py  - bash lib3d/run_tests.sh est
BiasVariance3D().reveal/fit(degree)/show_mean ; MLE3D().reveal/ascend ; Bayesian3D().reveal/update (x4)
KernelTrick3D().reveal/lift/unlift ; Smoothness3D().reveal/show_bumpy/regularize
RULES: reveal() sets its own camera ("oblique"); never design a scene that needs "front"/"side".
       core/guard.py auto-maps front/side -> slant. lint: bash lib3d/lint_views.sh

# lib3d/audio  (SFX from the real animation timeline, no music)
bash lib3d/audio/make_sfx.sh <scene.py> <SceneClass> <rendered.mp4>   -> <rendered>_sfx.mp4
timeline.py = TimelineMixin (logs play/move_camera) ; sfx.py = numpy sound designer (edit sounds there)
