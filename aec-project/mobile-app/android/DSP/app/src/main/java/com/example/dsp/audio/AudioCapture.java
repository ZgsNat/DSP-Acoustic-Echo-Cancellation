package com.example.dsp.audio;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Captures microphone audio at 16 kHz mono PCM16 and emits 1024-sample frames.
 */
public class AudioCapture {
    public static final int SAMPLE_RATE = 16000;
    public static final int FRAME_SIZE = 1024;

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private AudioRecord audioRecord;

    public interface FrameConsumer {
        void onFrame(short[] frame, long captureTimeNanos);
    }

    public synchronized void start(FrameConsumer consumer) {
        if (running.get()) {
            return;
        }

        int minBuffer = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        int targetBuffer = Math.max(minBuffer, FRAME_SIZE * 4);

        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                targetBuffer
        );

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            throw new IllegalStateException("AudioRecord is not initialized");
        }

        running.set(true);
        audioRecord.startRecording();

        executor.execute(() -> {
            short[] readBuffer = new short[FRAME_SIZE];
            while (running.get()) {
                int read = audioRecord.read(readBuffer, 0, FRAME_SIZE, AudioRecord.READ_BLOCKING);
                if (read <= 0) {
                    continue;
                }

                short[] frame = (read == FRAME_SIZE) ? readBuffer.clone() : Arrays.copyOf(readBuffer, FRAME_SIZE);
                consumer.onFrame(frame, System.nanoTime());
            }
        });
    }

    public synchronized void stop() {
        if (!running.get()) {
            return;
        }

        running.set(false);
        if (audioRecord != null) {
            try {
                audioRecord.stop();
            } catch (IllegalStateException ignored) {
                // No-op during shutdown races.
            }
            audioRecord.release();
            audioRecord = null;
        }
    }

    public boolean isRunning() {
        return running.get();
    }
}
