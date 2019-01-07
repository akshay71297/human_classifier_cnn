package com.akwares.classifier;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import com.wonderkiln.camerakit.CameraKit;
import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;


import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity{

    CameraView cameraView;
    Button cap;


    private static final int INPUT_SIZE = 300;
    private static final int PICK_IMAGE_REQUEST = 1;

    TensorFlowDetector tensorFlowClassifier;
    Bitmap bitmap;
    TextView txt;


    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cap = findViewById(R.id.cap);


        cameraView = findViewById(R.id.camera);
        cameraView.setFacing(CameraKit.Constants.FACING_FRONT);

        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {
            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                bitmap = Bitmap.createScaledBitmap(cameraKitImage.getBitmap(), INPUT_SIZE, INPUT_SIZE, false);

                //ImageView img = findViewById(R.id.imageView);

                //img.setImageBitmap(bitmap);



                final List<Recognition> results= tensorFlowClassifier.recognize(bitmap);

                Log.d("RSSS =", "= "+results.size());

                for (int i=0; i<results.size(); i++) {
                    txt.setText("title: "+results.get(0).getTitle()+" Confidence: "+results.get(0).getConfidence());
                    Log.i("id: " + results.get(i).getId(), "title: " + results.get(i).getTitle() + " Confidence: " + results.get(i).getConfidence());
                }
            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {
                Log.d("YESSSSSSSSSSS", "ENTROOOOOOOOOOOOOOOOOOOOOOOOO");

            }
        });


        cap.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage();
            }
        });



        txt = findViewById(R.id.txt1);
        tensorFlowClassifier = new TensorFlowDetector(getAssets());


        txt.setText("");

    }


    @Override
    protected void onPause() {
        super.onPause();
        cameraView.stop();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }



}
