package com.example.dsp;

import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import com.example.dsp.aec.AecBridge;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.switchmaterial.SwitchMaterial;
import com.google.android.material.textfield.TextInputEditText;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.util.Locale;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private TextInputEditText etPeerIp;
    private TextInputEditText etPeerPort;
    private TextInputEditText etListenPort;
    private MaterialButton btnStart;
    private MaterialButton btnStop;
    private SwitchMaterial switchAec;
    private TextView tvStatus;
    private TextView tvMetrics;

    private boolean callRunning = false;
    private final AecBridge aecBridge = new AecBridge();

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
        renderMetrics();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (callRunning) {
            stopCall();
        }
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

        aecBridge.initialize();
        aecBridge.setEnabled(switchAec.isChecked());

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

        callRunning = false;
        btnStart.setEnabled(true);
        btnStop.setEnabled(false);

        aecBridge.reset();
        setStatus("Status: Idle");
        renderMetrics();
    }

    private void renderMetrics() {
        Map<String, Double> m = aecBridge.getMetrics();
        double erle = valueOrZero(m, "erle_db");
        double delay = valueOrZero(m, "delay_ms");
        double dtd = valueOrZero(m, "double_talk_ratio");

        String text = String.format(
                Locale.US,
                "ERLE: %.2f dB\nDelay: %.2f ms\nDouble-talk: %.2f",
                erle,
                delay,
                dtd
        );
        tvMetrics.setText(text);
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
}