package com.example.dsp.audio;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;

import java.util.Arrays;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Plays incoming PCM16 frames and mirrors played frames to refQueue for AEC reference.
 */
public class AudioPlayback {
    public static final int SAMPLE_RATE = 16000;
    public static final int FRAME_SIZE = 1024;
    public static final int MAX_QUEUE_SIZE = 16;

    private final BlockingQueue<short[]> playQueue = new LinkedBlockingQueue<>(MAX_QUEUE_SIZE);
    private final BlockingQueue<short[]> refQueue = new LinkedBlockingQueue<>(MAX_QUEUE_SIZE);

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private AudioTrack audioTrack;
    private final short[] silence = new short[FRAME_SIZE];

    public synchronized void start() {
        if (running.get()) {
            return;
        }

        int minBuffer = AudioTrack.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );
        int targetBuffer = Math.max(minBuffer, FRAME_SIZE * 4);

        audioTrack = new AudioTrack(
                new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build(),
                new AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build(),
                targetBuffer,
                AudioTrack.MODE_STREAM,
                AudioManager.AUDIO_SESSION_ID_GENERATE
        );

        audioTrack.play();
        running.set(true);

        executor.execute(() -> {
            while (running.get()) {
                short[] frame = pollPlaybackFrame();
                pushReferenceFrame(frame);
                audioTrack.write(frame, 0, frame.length, AudioTrack.WRITE_BLOCKING);
            }
        });
    }

    public synchronized void stop() {
        if (!running.get()) {
            return;
        }

        running.set(false);
        playQueue.clear();
        refQueue.clear();

        if (audioTrack != null) {
            try {
                audioTrack.stop();
            } catch (IllegalStateException ignored) {
                // No-op during shutdown races.
            }
            audioTrack.release();
            audioTrack = null;
        }
    }

    public boolean enqueuePlaybackFrame(short[] frame) {
        if (frame == null || frame.length != FRAME_SIZE) {
            return false;
        }
        if (!playQueue.offer(frame.clone())) {
            playQueue.poll();
            return playQueue.offer(frame.clone());
        }
        return true;
    }

    public short[] pollReferenceFrameOrSilence() {
        short[] frame = refQueue.poll();
        return frame != null ? frame : Arrays.copyOf(silence, silence.length);
    }

    private short[] pollPlaybackFrame() {
        try {
            short[] frame = playQueue.poll(20, TimeUnit.MILLISECONDS);
            return frame != null ? frame : Arrays.copyOf(silence, silence.length);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Arrays.copyOf(silence, silence.length);
        }
    }

    private void pushReferenceFrame(short[] frame) {
        if (!refQueue.offer(frame.clone())) {
            refQueue.poll();
            refQueue.offer(frame.clone());
        }
    }
}
