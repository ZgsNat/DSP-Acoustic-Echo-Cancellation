package com.example.dsp.network;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Sends PCM16 frames over UDP with a 12-byte header compatible with desktop app.
 */
public class UdpSender {
    public static final int FRAME_SIZE = 1024;
    public static final int HEADER_SIZE = 12;
    public static final int MAX_QUEUE_SIZE = 32;

    private final BlockingQueue<short[]> sendQueue = new LinkedBlockingQueue<>(MAX_QUEUE_SIZE);
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicInteger sequence = new AtomicInteger(0);
    private final AtomicLong packetsSent = new AtomicLong(0);

    private volatile String lastError = "";

    private DatagramSocket socket;
    private InetAddress peerAddress;
    private int peerPort;

    public synchronized void start(String host, int port) throws IOException {
        if (running.get()) {
            return;
        }
        peerAddress = InetAddress.getByName(host);
        peerPort = port;
        socket = new DatagramSocket();
        lastError = "";
        running.set(true);

        executor.execute(() -> {
            while (running.get()) {
                try {
                    short[] frame = sendQueue.take();
                    sendPacket(frame);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (IOException e) {
                    // Network failures are tolerated; caller can inspect app status.
                    lastError = e.getClass().getSimpleName() + ": " + safeMessage(e);
                }
            }
        });
    }

    public synchronized void stop() {
        if (!running.get()) {
            return;
        }
        running.set(false);
        sendQueue.clear();
        if (socket != null) {
            socket.close();
            socket = null;
        }
    }

    public long getPacketsSent() {
        return packetsSent.get();
    }

    public String getLastError() {
        return lastError;
    }

    public boolean enqueueFrame(short[] frame) {
        if (frame == null || frame.length != FRAME_SIZE) {
            return false;
        }
        if (!sendQueue.offer(frame.clone())) {
            sendQueue.poll();
            return sendQueue.offer(frame.clone());
        }
        return true;
    }

    private void sendPacket(short[] frame) throws IOException {
        if (socket == null || peerAddress == null) {
            return;
        }

        byte[] payload = shortsToLittleEndianBytes(frame);
        int seq = sequence.getAndIncrement();
        int timestampMs = (int) (System.currentTimeMillis() & 0xFFFFFFFFL);

        ByteBuffer packetBuffer = ByteBuffer.allocate(HEADER_SIZE + payload.length).order(ByteOrder.BIG_ENDIAN);
        packetBuffer.putInt(seq);
        packetBuffer.putInt(timestampMs);
        packetBuffer.putInt(payload.length);
        packetBuffer.put(payload);

        DatagramPacket packet = new DatagramPacket(packetBuffer.array(), packetBuffer.array().length, peerAddress, peerPort);
        socket.send(packet);
        packetsSent.incrementAndGet();
    }

    private static String safeMessage(Exception e) {
        String message = e.getMessage();
        return message != null ? message : "no details";
    }

    private static byte[] shortsToLittleEndianBytes(short[] frame) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(frame.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (short sample : frame) {
            byteBuffer.putShort(sample);
        }
        return byteBuffer.array();
    }
}
