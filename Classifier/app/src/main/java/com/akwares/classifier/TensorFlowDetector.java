package com.akwares.classifier;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * Created by ak on 22/02/18.
 */

public class TensorFlowDetector {

    private static final int MAX_RESULTS = 100;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface;

    /*** API SHIT ***/
    private static String INPUT_NODE = "image_tensor";
    private static String[] OUTPUT_NODE =  new String[] {"detection_boxes", "detection_scores",
            "detection_classes", "num_detections"};
    private static final int INPUT_SIZE = 300;
    private static final String MODEL_FILE = "file:///android_asset/frozen_inference_graph.pb";
    private static final float MINIMUM_CONFIDENCE = 0.6f;


    private Vector<String> labels = new Vector<String>();

    /* FOR IMAGE RE Organizing*/
    int[] intValues = new int[INPUT_SIZE*INPUT_SIZE];
    private byte[] byteValues = new byte[INPUT_SIZE*INPUT_SIZE*3];
    private float[] outputLocations = new float[MAX_RESULTS*4];
    private float[] outputScores = new float[MAX_RESULTS];;
    private float[] outputClasses = new float[MAX_RESULTS];;
    private float[] outputNumDetections = new float[MAX_RESULTS];


    public TensorFlowDetector(AssetManager assetManager) {
        labels.add("akki");

        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
        Log.d("DEBUGGGGGGGGG", "model loaded successfully");

    }

    public static int argmax(float[] elems) {
        int bstIdx = 0;
        float bestV = elems[0];

        for(int i = 1; i < elems.length; i++){
            if(elems[i] > bestV){
                bestV = elems[i];
                bstIdx = i;
            }
        }

        return bstIdx;
    }

    private void img_to_bytes(Bitmap bitmap){
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
            byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
            byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
        }
    }

    public List<Recognition> recognize(final Bitmap bitmap){

        Log.d("heeeeeeeeeeeeeeeee", "weeeeeeeeeeeeeeeeeee");

        img_to_bytes(bitmap);

        inferenceInterface.feed(INPUT_NODE, byteValues, 1, INPUT_SIZE, INPUT_SIZE, 3);
        inferenceInterface.run(OUTPUT_NODE);

        outputLocations = new float[MAX_RESULTS * 4];
        outputScores = new float[MAX_RESULTS];
        outputClasses = new float[MAX_RESULTS];
        outputNumDetections = new float[1];
        inferenceInterface.fetch(OUTPUT_NODE[0], outputLocations);
        inferenceInterface.fetch(OUTPUT_NODE[1], outputScores);
        inferenceInterface.fetch(OUTPUT_NODE[2], outputClasses);
        inferenceInterface.fetch(OUTPUT_NODE[3], outputNumDetections);

        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        // Scale them back to the input size.
        for (int i = 0; i < outputScores.length; ++i) {
            final RectF detection =
                    new RectF(
                            outputLocations[4 * i + 1] * INPUT_SIZE,
                            outputLocations[4 * i] * INPUT_SIZE,
                            outputLocations[4 * i + 3] * INPUT_SIZE,
                            outputLocations[4 * i + 2] * INPUT_SIZE);

            if(outputScores[i] >= MINIMUM_CONFIDENCE) {
                pq.add(
                        new Recognition("" + i, labels.get(0)+i, outputScores[i], detection));
            }

        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

}
