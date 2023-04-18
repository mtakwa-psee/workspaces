#!/bin/bash
set -euo pipefail

# Variables
workdir="workdir"
device_workdir=/data/local/tmp/etienne/inception_v3_2
qnn_model_name="quantized"
QNN_TARGET_ARCH=aarch64-android

# Prepare Model Env
mkdir -p ${workdir}
#python3 ${QNN_SDK_ROOT}/examples/Models/InceptionV3/scripts/setup_inceptionv3.py -a ~/tmpdir -d ; true
#cp -r  ${QNN_SDK_ROOT}/examples/Models/InceptionV3/* ${workdir}/

# Convert qnn Model to onnx Model & Quantize it
${QNN_SDK_ROOT}/target/x86_64-linux-clang/bin/qnn-onnx-converter \
  --input_network ${workdir}/aligner_bundle/DisplacementEstimator_dsmv2_Aligner-orderly-masked-palm-civet-2-1e77.onnx \
  --output_path ${workdir}/aligner_bundle/DisplacementEstimator_dsmvxprior_Aligner-silent-cuscus-16-a9e9_htp_w8a8.cpp \
  --input_list /workspaces/work/workdir/qcalib_aligner_100/host_raw_list.txt \
  --use_per_channel_quantization true \
  --act_quantizer enhanced \
  --param_quantizer enhanced \
  --bias_bw 32 \
  --act_bw 8 \
  --algorithms cle

# Compile the QNN Model
${QNN_SDK_ROOT}/target/x86_64-linux-clang/bin/qnn-model-lib-generator \
  -c ${workdir}/aligner_bundle/${qnn_model_name}.cpp \
  -b ${workdir}/aligner_bundle/${qnn_model_name}.bin \
  -o ${workdir}/libs

# Serialize the QNN Model for running on HTP
${QNN_SDK_ROOT}/target/x86_64-linux-clang/bin/qnn-context-binary-generator \
              --backend ${QNN_SDK_ROOT}/target/x86_64-linux-clang/lib/libQnnHtp.so \
              --model ${workdir}/libs/x86_64-linux-clang/lib${qnn_model_name}.so \
              --binary_file ${qnn_model_name}.serialized \
              --output_dir ${workdir}/libs

# Create Execution Context On-device
adb shell "mkdir -p ${device_workdir}"
adb push ${QNN_SDK_ROOT}/target/hexagon-v69/lib/unsigned/libQnnHtpV69Skel.so ${device_workdir}
adb push ${QNN_SDK_ROOT}/target/${QNN_TARGET_ARCH}/lib/libQnnHtpV69Stub.so ${device_workdir}
adb push ${QNN_SDK_ROOT}/target/${QNN_TARGET_ARCH}/lib/libQnnHtp.so ${device_workdir}
adb push ${QNN_SDK_ROOT}/target/${QNN_TARGET_ARCH}/bin/qnn-net-run ${device_workdir}

adb push ${workdir}/libs/${qnn_model_name}.serialized.bin ${device_workdir}
adb push ${workdir}/data/cropped ${device_workdir}
adb push ${workdir}/data/target_raw_list.txt ${device_workdir}

# Login to Device and Exec Model Inference on HTP
adb shell """
cd ${device_workdir}
ls -lh
export LD_LIBRARY_PATH=${device_workdir}
export ADSP_LIBRARY_PATH="${device_workdir}"
./qnn-net-run --backend libQnnHtp.so --input_list target_raw_list.txt --retrieve_context ${qnn_model_name}.serialized.bin
ls output
"""

# Get Results Back to Host
adb pull /data/local/tmp/inception_v3/output ${workdir}/output_android

# Do Fancy labeling check
python3 ${QNN_SDK_ROOT}/examples/Models/InceptionV3/scripts/show_inceptionv3_classifications.py \
  -i ${workdir}/data/cropped/raw_list.txt \
  -o ${workdir}/output_android/ \
  -l ${workdir}/data/imagenet_slim_labels.txt
  




    #--out_node InceptionV3/Predictions/Reshape_1 \
  #--output_path ${workdir}/model/${qnn_model_name}.cpp \