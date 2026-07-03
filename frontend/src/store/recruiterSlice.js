/**
 * recruiterSlice.js
 *
 * Manages the recruiter batch-analysis flow:
 *   resumes (File[]), jdFile → loading → data | error
 */
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

const API_BATCH = `${import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"}/api/v1/recruiter/batch-analyze`;

// ── Async thunk ──────────────────────────────────────────────────────────────

/**
 * Dispatch with `{ files: File[], jdFile: File, token: string, companyName: string }`.
 */
export const runBatchAnalysis = createAsyncThunk(
    "recruiter/batchAnalyze",
    async ({ files, jdFile, token, companyName }, { rejectWithValue }) => {
        const form = new FormData();
        files.forEach((f) => form.append("resumes", f));
        form.append("job_description_file", jdFile);

        const res = await fetch(API_BATCH, {
            method: "POST",
            headers: {
                Authorization: `Bearer ${token}`,
                "X-Company-Name": companyName,
            },
            body: form,
        });

        const out = await res.json().catch(() => ({}));
        if (!res.ok) {
            return rejectWithValue(out.detail || "Batch analyze failed.");
        }
        return out;
    }
);

// ── Slice ────────────────────────────────────────────────────────────────────

const recruiterSlice = createSlice({
    name: "recruiter",
    initialState: {
        /** Array of File objects (resume uploads). */
        files: [],
        /** Single File object (job description). */
        jdFile: null,
        /** "idle" | "loading" | "success" | "error" */
        status: "idle",
        data: null,
        error: "",
    },
    reducers: {
        setFiles(state, { payload }) {
            state.files = payload;
            state.error = "";
        },
        setJdFile(state, { payload }) {
            state.jdFile = payload;
            state.error = "";
        },
        clearRecruiterResult(state) {
            state.data = null;
            state.status = "idle";
            state.error = "";
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(runBatchAnalysis.pending, (state) => {
                state.status = "loading";
                state.error = "";
                state.data = null;
            })
            .addCase(runBatchAnalysis.fulfilled, (state, { payload }) => {
                state.status = "success";
                state.data = payload;
            })
            .addCase(runBatchAnalysis.rejected, (state, { payload }) => {
                state.status = "error";
                state.error = payload ?? "An unexpected error occurred.";
            });
    },
});

export const { setFiles, setJdFile, clearRecruiterResult } = recruiterSlice.actions;

// ── Selectors ────────────────────────────────────────────────────────────────

export const selectRecruiterFiles = (s) => s.recruiter.files;
export const selectJdFile = (s) => s.recruiter.jdFile;
export const selectRecruiterStatus = (s) => s.recruiter.status;
export const selectRecruiterData = (s) => s.recruiter.data;
export const selectRecruiterError = (s) => s.recruiter.error;
export const selectRecruiterLoading = (s) => s.recruiter.status === "loading";

export default recruiterSlice.reducer;
