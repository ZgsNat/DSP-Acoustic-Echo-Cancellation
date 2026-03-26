package com.example.dsp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Patterns;
import android.widget.TextView;
import android.widget.Toast;

import com.example.dsp.aec.AecBridge;
import com.example.dsp.audio.AudioCapture;
import com.example.dsp.audio.AudioPlayback;
import com.example.dsp.network.UdpReceiver;
import com.example.dsp.network.UdpSender;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.textfield.TextInputEditText;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.IOException;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_RECORD_AUDIO = 1001;

    private TextInputEditText etPeerIp;
    private TextInputEditText etPeerPort;
    private TextInputEditText etListenPort;
    private MaterialButton btnStart;
    private MaterialButton btnStop;
    private SwitchMaterial switchAec;
    private TextView tvStatus;
    private TextView tvMetrics;
    private TextView tvLocalIp;

    private boolean callRunning = false;
    private final AecBridge aecBridge = new AecBridge();

    private final AudioCapture audioCapture = new AudioCapture();
    private final AudioPlayback audioPlayback = new AudioPlayback();
    private final UdpSender udpSender = new UdpSender();
    private final UdpReceiver udpReceiver = new UdpReceiver();

    private final AtomicBoolean bridgeLoopRunning = new AtomicBoolean(false);
    private final ExecutorService receiverBridgeExecutor = Executors.newSingleThreadExecutor();
    private final ScheduledExecutorService uiStatsExecutor = Executors.newSingleThreadScheduledExecutor();

    private String lastPeerIp = "";
    private int lastPeerPort = 0;
    private int lastListenPort = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        bindViews();
        bindActions();
        displayLocalIpAddress();
        ensureRecordAudioPermission();
        startUiStatsLoop();
        renderMetrics();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (callRunning) {
            stopCall();
        }
        uiStatsExecutor.shutdownNow();
        receiverBridgeExecutor.shutdownNow();
        aecBridge.release();
    }

    private void bindViews() {
        etPeerIp = findViewById(R.id.etPeerIp);
        etPeerPort = findViewById(R.id.etPeerPort);
        etListenPort = findViewById(R.id.etListenPort);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);
        switchAec = findViewById(R.id.switchAec);
        tvStatus = findViewById(R.id.tvStatus);
        tvMetrics = findViewById(R.id.tvMetrics);
        tvLocalIp = findViewById(R.id.tvLocalIp);
    }

    private void bindActions() {
        btnStart.setOnClickListener(v -> {
            if (!validateInputs()) {
                return;
            }
            startCall();
        });

        btnStop.setOnClickListener(v -> stopCall());

        switchAec.setOnCheckedChangeListener((buttonView, isChecked) -> {
            aecBridge.setEnabled(isChecked);
            String state = isChecked ? "ON" : "OFF";
            if (callRunning) {
                setStatus("Status: Running | AEC " + state);
            }
            renderMetrics();
        });
    }

    private boolean validateInputs() {
        String peerIp = textOf(etPeerIp);
        String peerPortText = textOf(etPeerPort);
        String listenPortText = textOf(etListenPort);

        if (peerIp.isEmpty()) {
            toast("Peer IP is required");
            return false;
        }

        if (!isValidIpv4(peerIp)) {
            toast("Peer IP must be a valid IPv4 address, e.g. 192.168.1.10");
            return false;
        }

        Integer peerPort = parsePort(peerPortText);
        Integer listenPort = parsePort(listenPortText);
        if (peerPort == null || listenPort == null) {
            toast("Ports must be between 1 and 65535");
            return false;
        }

        return true;
    }

    private void startCall() {
        if (callRunning) {
            return;
        }

        if (!hasRecordAudioPermission()) {
            ensureRecordAudioPermission();
            toast("Microphone permission is required");
            return;
        }

        String peerIp = textOf(etPeerIp);
        Integer peerPort = parsePort(textOf(etPeerPort));
        Integer listenPort = parsePort(textOf(etListenPort));
        if (peerPort == null || listenPort == null) {
            toast("Ports must be between 1 and 65535");
            return;
        }

        lastPeerIp = peerIp;
        lastPeerPort = peerPort;
        lastListenPort = listenPort;

        aecBridge.initialize();
        aecBridge.setEnabled(switchAec.isChecked());

        try {
            udpReceiver.start(listenPort);
            audioPlayback.start();
            udpSender.start(peerIp, peerPort);
            audioCapture.start((micFrame, captureTimeNanos) -> {
                short[] refFrame = audioPlayback.pollReferenceFrameOrSilence();
                short[] outFrame = aecBridge.processFrame(micFrame, refFrame);
                udpSender.enqueueFrame(outFrame);
            });
        } catch (IOException | IllegalStateException e) {
            stopRealtimePipeline();
            setStatus("Status: Error starting call");
            toast(buildStartupErrorMessage(e));
            return;
        }

        startReceiverToPlaybackBridge();

        callRunning = true;
        btnStart.setEnabled(false);
        btnStop.setEnabled(true);
        switchAec.setEnabled(true);

        String state = switchAec.isChecked() ? "ON" : "OFF";
        setStatus("Status: Running | AEC " + state);
        renderMetrics();
    }

    private void stopCall() {
        if (!callRunning) {
            return;
        }

        stopRealtimePipeline();

        callRunning = false;
        btnStart.setEnabled(true);
        btnStop.setEnabled(false);

        aecBridge.reset();
        setStatus("Status: Idle");
        renderMetrics();
    }

    private void startReceiverToPlaybackBridge() {
        bridgeLoopRunning.set(true);
        receiverBridgeExecutor.execute(() -> {
            while (bridgeLoopRunning.get()) {
                short[] frame = udpReceiver.takeFrameOrSilence(80);
                audioPlayback.enqueuePlaybackFrame(frame);
            }
        });
    }

    private void startUiStatsLoop() {
        uiStatsExecutor.scheduleAtFixedRate(() -> runOnUiThread(this::renderMetrics),
                0,
                500,
                TimeUnit.MILLISECONDS);
    }

    private void stopRealtimePipeline() {
        bridgeLoopRunning.set(false);
        audioCapture.stop();
        udpSender.stop();
        udpReceiver.stop();
        audioPlayback.stop();
    }

    private boolean hasRecordAudioPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void ensureRecordAudioPermission() {
        if (!hasRecordAudioPermission()) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    REQUEST_RECORD_AUDIO
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.length == 0 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                toast("Microphone permission denied");
            }
        }
    }

    private void renderMetrics() {
        Map<String, Double> m = aecBridge.getMetrics();
        double erle = valueOrZero(m, "erle_db");
        double delay = valueOrZero(m, "delay_ms");
        double dtd = valueOrZero(m, "double_talk_ratio");

        long tx = udpSender.getPacketsSent();
        long rx = udpReceiver.getPacketsReceived();
        long loss = udpReceiver.getPacketsLost();

        String senderError = udpSender.getLastError();
        String receiverError = udpReceiver.getLastError();
        String networkError = "";
        if (!senderError.isEmpty()) {
            networkError = "\nTX error: " + senderError;
        }
        if (!receiverError.isEmpty()) {
            networkError += "\nRX error: " + receiverError;
        }

        String text = String.format(
                Locale.US,
                "ERLE: %.2f dB\nDelay: %.2f ms\nDouble-talk: %.2f\nTX: %d  RX: %d  Loss: %d%s",
                erle,
                delay,
                dtd,
                tx,
                rx,
                loss,
                networkError
        );
        tvMetrics.setText(text);

        if (callRunning) {
            String state = switchAec.isChecked() ? "ON" : "OFF";
            String status = String.format(
                    Locale.US,
                    "Status: Running | AEC %s | %s:%d -> listen %d",
                    state,
                    lastPeerIp,
                    lastPeerPort,
                    lastListenPort
            );
            setStatus(status);
        }
    }

    private static double valueOrZero(Map<String, Double> metrics, String key) {
        if (metrics == null || !metrics.containsKey(key) || metrics.get(key) == null) {
            return 0.0;
        }
        return metrics.get(key);
    }

    private void setStatus(String status) {
        tvStatus.setText(status);
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    private static String textOf(TextInputEditText editText) {
        if (editText.getText() == null) {
            return "";
        }
        return editText.getText().toString().trim();
    }

    private static Integer parsePort(String raw) {
        try {
            int value = Integer.parseInt(raw);
            if (value < 1 || value > 65535) {
                return null;
            }
            return value;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static boolean isValidIpv4(String ip) {
        if (!Patterns.IP_ADDRESS.matcher(ip).matches()) {
            return false;
        }

        String[] parts = ip.split("\\.");
        if (parts.length != 4) {
            return false;
        }

        for (String part : parts) {
            if (part.isEmpty()) {
                return false;
            }
            int value;
            try {
                value = Integer.parseInt(part);
            } catch (NumberFormatException e) {
                return false;
            }
            if (value < 0 || value > 255) {
                return false;
            }
        }

        return true;
    }

    private String buildStartupErrorMessage(Exception e) {
        String detail = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
        return "Start failed: " + detail;
    }

    private void displayLocalIpAddress() {
        new Thread(() -> {
            final String localIp = getLocalIpAddress();
            runOnUiThread(() -> {
                if (tvLocalIp != null) {
                    tvLocalIp.setText("This Device IP: " + localIp);
                }
            });
        }).start();
    }

    private String getLocalIpAddress() {
        try {
            for (NetworkInterface intf : Collections.list(NetworkInterface.getNetworkInterfaces())) {
                for (InetAddress addr : Collections.list(intf.getInetAddresses())) {
                    if (!addr.isLoopbackAddress()) {
                        String hostAddr = addr.getHostAddress();
                        // IPv4 addresses only (without %)
                        if (hostAddr.indexOf(':') < 0) {
                            return hostAddr;
                        }
                    }
                }
            }
        } catch (Exception ignored) {
            // Fall through to default
        }
        return "(unknown)";
    }
}