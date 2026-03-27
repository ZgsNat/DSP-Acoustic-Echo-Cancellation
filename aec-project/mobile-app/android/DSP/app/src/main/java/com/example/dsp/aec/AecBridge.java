package com.example.dsp.aec;

import java.lang.reflect.Array;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

/**
 * Java-side bridge skeleton for Python AECPipeline integration (Chaquopy planned).
 *
 * Current behavior:
 * - If bridge is not initialized, processFrame returns raw mic frame (bypass mode).
 * - Methods are thread-safe to simplify early integration with audio threads.
 */
public class AecBridge {
    public static final int FRAME_SIZE = 1024;
    private static final float PCM16_SCALE = 32768.0f;

    private boolean initialized = false;
    private boolean enabled = false;

    private boolean pythonAvailable = false;

    // Reflection handles to avoid hard dependency when Chaquopy is not configured yet.
    private Object pythonModule;
    private Method moduleCallAttr;

    private Object pythonBridgeHandle;

    public synchronized void initialize() {
        if (initialized) {
            return;
        }

        try {
            Class<?> pythonClass = Class.forName("com.chaquo.python.Python");
            Method getInstance = pythonClass.getMethod("getInstance");
            Method getModule = pythonClass.getMethod("getModule", String.class);

            Object pythonInstance = getInstance.invoke(null);
            Object module = getModule.invoke(pythonInstance, "aec_android_bridge");

            Class<?> pyObjectClass = Class.forName("com.chaquo.python.PyObject");
            Method callAttr = pyObjectClass.getMethod("callAttr", String.class, Object[].class);

            // Optional initialization hook on Python side.
            invokeModuleMethod(module, callAttr, "init_pipeline");

            this.pythonModule = module;
            this.moduleCallAttr = callAttr;
            this.pythonBridgeHandle = module;
            this.pythonAvailable = true;
        } catch (ClassNotFoundException
                | NoSuchMethodException
                | IllegalAccessException
                | InvocationTargetException e) {
            this.pythonModule = null;
            this.moduleCallAttr = null;
            this.pythonBridgeHandle = null;
            this.pythonAvailable = false;
        }

        initialized = true;
    }

    public synchronized void setEnabled(boolean value) {
        enabled = value;
    }

    public synchronized boolean isEnabled() {
        return enabled;
    }

    public synchronized short[] processFrame(short[] micFrame, short[] refFrame) {
        if (micFrame == null || micFrame.length != FRAME_SIZE) {
            throw new IllegalArgumentException("micFrame must be 1024 samples");
        }
        if (refFrame == null || refFrame.length != FRAME_SIZE) {
            throw new IllegalArgumentException("refFrame must be 1024 samples");
        }

        if (!initialized || !enabled) {
            return micFrame.clone();
        }

        if (!pythonAvailable || pythonModule == null || moduleCallAttr == null) {
            return micFrame.clone();
        }

        try {
            float[] micFloat = pcm16ToFloat32(micFrame);
            float[] refFloat = pcm16ToFloat32(refFrame);

            Object pyResult = invokeModuleMethod(pythonModule, moduleCallAttr, "process_frame", micFloat, refFloat);
            float[] cleanFloat = toFloatArray(pyResult);

            if (cleanFloat == null || cleanFloat.length == 0) {
                return micFrame.clone();
            }

            if (cleanFloat.length != FRAME_SIZE) {
                float[] resized = new float[FRAME_SIZE];
                System.arraycopy(cleanFloat, 0, resized, 0, Math.min(cleanFloat.length, FRAME_SIZE));
                cleanFloat = resized;
            }

            return float32ToPcm16(cleanFloat);
        } catch (Exception e) {
            // Bridge failures should not break call flow.
            return micFrame.clone();
        }
    }

    public synchronized Map<String, Double> getMetrics() {
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("erle_db", 0.0);
        metrics.put("delay_ms", 0.0);
        metrics.put("double_talk_ratio", 0.0);

        if (!initialized) {
            return metrics;
        }

        if (!pythonAvailable || pythonModule == null || moduleCallAttr == null) {
            return metrics;
        }

        try {
            Object pyMetrics = invokeModuleMethod(pythonModule, moduleCallAttr, "get_metrics");
            if (pyMetrics == null) {
                return metrics;
            }

            putMetricIfPresent(metrics, "erle_db", extractPyDictValue(pyMetrics, "erle_db"));
            putMetricIfPresent(metrics, "delay_ms", extractPyDictValue(pyMetrics, "delay_ms"));
            putMetricIfPresent(metrics, "double_talk_ratio", extractPyDictValue(pyMetrics, "double_talk_ratio"));
            putMetricIfPresent(metrics, "filter_norm", extractPyDictValue(pyMetrics, "filter_norm"));
        } catch (Exception ignored) {
            // Keep default metrics on bridge errors.
        }

        return metrics;
    }

    public synchronized void reset() {
        if (!initialized) {
            return;
        }

        if (!pythonAvailable || pythonModule == null || moduleCallAttr == null) {
            return;
        }

        try {
            invokeModuleMethod(pythonModule, moduleCallAttr, "reset");
        } catch (Exception ignored) {
            // No-op to preserve realtime stability.
        }
    }

    public synchronized void release() {
        enabled = false;
        initialized = false;
        pythonAvailable = false;
        pythonModule = null;
        moduleCallAttr = null;
        pythonBridgeHandle = null;
    }

    private static float[] pcm16ToFloat32(short[] pcm) {
        float[] out = new float[pcm.length];
        for (int i = 0; i < pcm.length; i++) {
            out[i] = pcm[i] / PCM16_SCALE;
        }
        return out;
    }

    private static short[] float32ToPcm16(float[] audio) {
        short[] out = new short[audio.length];
        for (int i = 0; i < audio.length; i++) {
            float v = Math.max(-1.0f, Math.min(1.0f, audio[i]));
            out[i] = (short) Math.round(v * 32767.0f);
        }
        return out;
    }

    private static Object invokeModuleMethod(Object module, Method callAttr, String name, Object... args)
            throws InvocationTargetException, IllegalAccessException {
        return callAttr.invoke(module, name, args);
    }

    private static float[] toFloatArray(Object pyResult) {
        if (pyResult == null) {
            return null;
        }

        try {
            Method toJava = pyResult.getClass().getMethod("toJava", Class.class);

            Object primitiveArray = toJava.invoke(pyResult, float[].class);
            if (primitiveArray instanceof float[]) {
                return (float[]) primitiveArray;
            }

            Object boxedArray = toJava.invoke(pyResult, Float[].class);
            if (boxedArray instanceof Float[]) {
                Float[] boxed = (Float[]) boxedArray;
                float[] out = new float[boxed.length];
                for (int i = 0; i < boxed.length; i++) {
                    out[i] = boxed[i] != null ? boxed[i] : 0.0f;
                }
                return out;
            }
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException ignored) {
            // Fall through to null.
        }

        return null;
    }

    private static Object extractPyDictValue(Object pyDict, String key)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Method getItem = pyDict.getClass().getMethod("getItem", Object.class);
        return getItem.invoke(pyDict, key);
    }

    private static void putMetricIfPresent(Map<String, Double> metrics, String key, Object pyValue) {
        Double value = toDouble(pyValue);
        if (value != null) {
            metrics.put(key, value);
        }
    }

    private static Double toDouble(Object pyValue) {
        if (pyValue == null) {
            return null;
        }

        if (pyValue instanceof Number) {
            return ((Number) pyValue).doubleValue();
        }

        try {
            Method toJava = pyValue.getClass().getMethod("toJava", Class.class);
            Object converted = toJava.invoke(pyValue, Double.class);
            if (converted instanceof Double) {
                return (Double) converted;
            }
            converted = toJava.invoke(pyValue, Float.class);
            if (converted instanceof Float) {
                return ((Float) converted).doubleValue();
            }
            converted = toJava.invoke(pyValue, Integer.class);
            if (converted instanceof Integer) {
                return ((Integer) converted).doubleValue();
            }
            converted = toJava.invoke(pyValue, Long.class);
            if (converted instanceof Long) {
                return ((Long) converted).doubleValue();
            }
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException ignored) {
            // Return null if conversion is unavailable.
        }

        return null;
    }
}
