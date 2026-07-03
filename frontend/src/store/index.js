/**
 * store/index.js — Redux store configuration
 *
 * Slices:
 *   analysis   → user-facing resume analysis (Dashboard.jsx)
 *   recruiter  → recruiter batch shortlist (recruiter-dashboard.jsx)
 *
 * Redux DevTools Extension works out-of-the-box in development mode.
 * In production the devTools middleware is stripped automatically by RTK.
 */
import { configureStore } from "@reduxjs/toolkit";
import analysisReducer from "./analysisSlice";
import recruiterReducer from "./recruiterSlice";

const store = configureStore({
    reducer: {
        analysis: analysisReducer,
        recruiter: recruiterReducer,
    },
    // RTK's default middleware already includes redux-thunk and
    // serializability checks. We loosen the check for File objects (which
    // are not plain-serialisable) stored in state.analysis.file and
    // state.recruiter.files / state.recruiter.jdFile.
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            serializableCheck: {
                ignoredPaths: [
                    "analysis.file",
                    "recruiter.files",
                    "recruiter.jdFile",
                ],
            },
        }),
});

export default store;
