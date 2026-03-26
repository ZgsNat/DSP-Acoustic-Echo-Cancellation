package com.example.dsp.network;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Receives UDP audio packets, decodes PCM16 frames, and exposes frames for playback queue.
 */
public class UdpReceiver {
    public static final int FRAME_SIZE = 1024;
    public static final int HEADER_SIZE = 12;
    public static final int MAX_PACKET_SIZE = HEADER_SIZE + FRAME_SIZE * 2 + 64;
    public static final int MAX_QUEUE_SIZE = 32;

    private final BlockingQueue<short[]> receiveQueue = new LinkedBlockingQueue<>(MAX_QUEUE_SIZE);
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicLong packetsReceived = new AtomicLong(0);
    private final AtomicLong packetsLost = new AtomicLong(0);

    private DatagramSocket socket;
    private Integer expectedSequence;
    private volatile String lastError = "";

    public synchronized void start(int listenPort) throws IOException {
        if (running.get()) {
            return;
        }

        socket = new DatagramSocket(listenPort);
        socket.setSoTimeout(500);
        expectedSequence = null;
        lastError = "";
        running.set(true);

        executor.execute(() -> {
            byte[] buffer = new byte[MAX_PACKET_SIZE];
            while (running.get()) {
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                try {
                    socket.receive(packet);
                    onPacket(packet.getData(), packet.getLength());
                } catch (SocketTimeoutException ignored) {
                    // Timeout allows cooperative stop checks.
                } catch (IOException e) {
                    lastError = e.getClass().getSimpleName() + ": " + safeMessage(e);
                    if (!running.get()) {
                        break;
                    }
                }
            }
        });
    }

    public synchronized void stop() {
        if (!running.get()) {
            return;
        }
        running.set(false);
        receiveQueue.clear();
        if (socket != null) {
            socket.close();
            socket = null;
        }
    }

    public short[] pollFrameOrSilence() {
        short[] frame = receiveQueue.poll();
        return frame != null ? frame : new short[FRAME_SIZE];
    }

    public short[] takeFrameOrSilence(long timeoutMs) {
        try {
            short[] frame = receiveQueue.poll(timeoutMs, TimeUnit.MILLISECONDS);
            return frame != null ? frame : new short[FRAME_SIZE];
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new short[FRAME_SIZE];
        }
    }

    public long getPacketsReceived() {
        return packetsReceived.get();
    }

    public long getPacketsLost() {
        return packetsLost.get();
    }

    public String getLastError() {
        return lastError;
    }

    private void onPacket(byte[] data, int length) {
        if (length < HEADER_SIZE) {
            return;
        }

        ByteBuffer header = ByteBuffer.wrap(data, 0, HEADER_SIZE).order(ByteOrder.BIG_ENDIAN);
        int seq = header.getInt();
        header.getInt(); // timestamp_ms
        int payloadLength = header.getInt();

        if (payloadLength <= 0 || HEADER_SIZE + payloadLength > length) {
            return;
        }

        handleSequenceGap(seq);

        byte[] payload = Arrays.copyOfRange(data, HEADER_SIZE, HEADER_SIZE + payloadLength);
        short[] frame = littleEndianBytesToShorts(payload);
        if (frame.length != FRAME_SIZE) {
            frame = Arrays.copyOf(frame, FRAME_SIZE);
        }

        packetsReceived.incrementAndGet();

        if (!receiveQueue.offer(frame)) {
            receiveQueue.poll();
            receiveQueue.offer(frame);
        }
    }

    private void handleSequenceGap(int seq) {
        if (expectedSequence == null) {
            expectedSequence = seq;
        }
        long gap = (seq - expectedSequence) & 0xFFFFFFFFL;
        if (gap > 0 && gap < 100) {
            packetsLost.addAndGet(gap);
            for (int i = 0; i < Math.min(gap, 4); i++) {
                short[] silence = new short[FRAME_SIZE];
                if (!receiveQueue.offer(silence)) {
                    receiveQueue.poll();
                    receiveQueue.offer(silence);
                }
            }
        }
        expectedSequence = seq + 1;
    }

    private static short[] littleEndianBytesToShorts(byte[] payload) {
        int samples = payload.length / 2;
        short[] out = new short[samples];
        ByteBuffer bb = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < samples; i++) {
            out[i] = bb.getShort();
        }
        return out;
    }

    private static String safeMessage(Exception e) {
        String message = e.getMessage();
        return message != null ? message : "no details";
    }
}
