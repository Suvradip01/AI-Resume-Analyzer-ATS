/**
 * analysisSlice.js
 *
 * Manages the full user-facing resume analysis flow:
 *   file, jobDescription → loading → result | error
 *
 * The `runAnalysis` thunk handles FormData construction and the API call
 * so the component layer stays pure UI.
 */
import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";

const API_URL = `${import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"}/api/v1/resume/analyze`;

// ── Async thunk ──────────────────────────────────────────────────────────────

/**
 * Dispatch this with `{ file: File, jobDescription: string }`.
 *
 * The thunk serialises the file into FormData, calls the backend, and
 * returns the parsed JSON on success (or rejects with a human-readable
 * error message that the slice stores in `state.error`).
 */
export const runAnalysis = createAsyncThunk(
    "analysis/run",
    async ({ file, jobDescription }, { rejectWithValue }) => {
        const formData = new FormData();
        formData.append("resume_file", file);
        formData.append(
            "job_description",
            JSON.stringify({ title: "Target Role", description: jobDescription })
        );

        const response = await fetch(API_URL, { method: "POST", body: formData });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            return rejectWithValue(err.detail || "Analysis failed. Please try again.");
        }

        return await response.json();
    }
);

// ── Slice ────────────────────────────────────────────────────────────────────

const analysisSlice = createSlice({
    name: "analysis",
    initialState: {
        /** The File object selected by the user (not serialisable — stored by reference). */
        file: null,
        jobDescription: "",
        /** "idle" | "loading" | "success" | "error" */
        status: "idle",
        result: null,
        error: "",
    },
    reducers: {
        setFile(state, { payload }) {
            state.file = payload;
            // Clear any previous error when the user picks a new file.
            state.error = "";
        },
        setJobDescription(state, { payload }) {
            state.jobDescription = payload;
        },
        clearResult(state) {
            state.result = null;
            state.status = "idle";
            state.error = "";
        },
        clearError(state) {
            state.error = "";
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(runAnalysis.pending, (state) => {
                state.status = "loading";
                state.error = "";
                state.result = null;
            })
            .addCase(runAnalysis.fulfilled, (state, { payload }) => {
                state.status = "success";
                state.result = payload;
            })
            .addCase(runAnalysis.rejected, (state, { payload }) => {
                state.status = "error";
                state.error = payload ?? "An unexpected error occurred.";
            });
    },
});

export const { setFile, setJobDescription, clearResult, clearError } = analysisSlice.actions;

// ── Selectors ────────────────────────────────────────────────────────────────

export const selectFile = (s) => s.analysis.file;
export const selectJobDescription = (s) => s.analysis.jobDescription;
export const selectAnalysisStatus = (s) => s.analysis.status;
export const selectResult = (s) => s.analysis.result;
export const selectError = (s) => s.analysis.error;
export const selectIsLoading = (s) => s.analysis.status === "loading";

export default analysisSlice.reducer;
