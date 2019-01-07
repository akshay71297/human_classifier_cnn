package com.akwares.classifier;

import android.graphics.Rect;
import android.graphics.RectF;

/**
 * Created by ak on 06/03/18.
 */

public class Recognition {


    private final String id;
    private final String title;
    private final float confidence;
    private RectF location;

    public Recognition(String id, String title, float confidence, RectF location) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public float getConfidence() {
        return confidence;
    }

    public void setLocation(RectF location) {
        this.location = location;
    }

    public RectF getLocation() {
        return location;
    }


}
