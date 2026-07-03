import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import {
    Upload, FileText, LogOut, Home, Trophy,
    Award, Medal, CheckCircle2, AlertTriangle, Info, ChevronRight,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth, useUser, useClerk } from "@clerk/clerk-react";

import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";

import {
    setFiles,
    setJdFile,
    runBatchAnalysis,
    selectRecruiterFiles,
    selectJdFile,
    selectRecruiterLoading,
    selectRecruiterData,
    selectRecruiterError,
} from "../store/recruiterSlice";

const MotionDiv = motion.div;

// ── Pure helpers ─────────────────────────────────────────────────────────────

const getRankIcon = (rank) => {
    switch (rank) {
        case 1: return <Award className="size-6 text-yellow-400 drop-shadow-[0_0_10px_rgba(250,204,21,0.5)]" />;
        case 2: return <Medal className="size-6 text-neutral-300 drop-shadow-[0_0_10px_rgba(212,212,216,0.5)]" />;
        case 3: return <Medal className="size-6 text-amber-700 drop-shadow-[0_0_10px_rgba(180,83,9,0.5)]" />;
        default: return <span className="text-sm font-bold text-neutral-500">#{rank}</span>;
    }
};

const getRankStyle = (rank) => {
    switch (rank) {
        case 1: return "bg-gradient-to-br from-yellow-500/10 to-yellow-900/5 border-yellow-500/30";
        case 2: return "bg-gradient-to-br from-neutral-300/10 to-neutral-700/5 border-neutral-400/30";
        case 3: return "bg-gradient-to-br from-amber-700/10 to-amber-900/5 border-amber-700/30";
        default: return "bg-neutral-900/40 border-white/5";
    }
};

const cleanFeedbackText = (text) => {
    if (!text) return "";
    let cleaned = text.replace(/[\uFFFD\u203D◆♦]/g, "");
    const levelMatch = cleaned.match(/\((Basic|Medium|Advanced)\s*Level\)/i);
    if (levelMatch) {
        cleaned = cleaned.replace(/^.*?(\((Basic|Medium|Advanced)\s*Level\)).*?:\s*/i, "$1. ");
    } else {
        const labelPattern = /^[\s\W]*(\*\*)?(Skills|Experience Relevance|Experience|Project Detected|Project|Overall|Tip for [^:]+|Final Verdict)(\*\*)?:\s*/i;
        cleaned = cleaned.replace(labelPattern, "");
    }
    cleaned = cleaned.replace(/^[^a-zA-Z0-9*(\"'`[\]]+/g, "");
    cleaned = cleaned.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>');
    return cleaned.trim();
};

const getFeedbackIcon = (text) => {
    if (text.includes("✅")) return <CheckCircle2 className="size-3.5 text-green-500 shrink-0 mt-0.5" />;
    if (text.includes("❌")) return <AlertTriangle className="size-3.5 text-red-500 shrink-0 mt-0.5" />;
    return <Info className="size-3.5 text-blue-400 shrink-0 mt-0.5" />;
};

const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } },
};

const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function RecruiterDashboardPage() {
    const nav      = useNavigate();
    const dispatch = useDispatch();

    const { isLoaded, isSignedIn, getToken } = useAuth();
    const { user }  = useUser();
    const { signOut } = useClerk();

    // Redux state
    const files   = useSelector(selectRecruiterFiles);
    const jdFile  = useSelector(selectJdFile);
    const loading = useSelector(selectRecruiterLoading);
    const data    = useSelector(selectRecruiterData);
    const error   = useSelector(selectRecruiterError);

    useEffect(() => {
        if (isLoaded && !isSignedIn) nav("/recruiter");
    }, [isLoaded, isSignedIn, nav]);

    async function logout() {
        await signOut();
        nav("/recruiter");
    }

    async function analyze() {
        if (!files.length || !jdFile) return;

        let token;
        try {
            token = await getToken();
        } catch {
            nav("/recruiter");
            return;
        }
        if (!token) { nav("/recruiter"); return; }

        const companyName = user?.unsafeMetadata?.company || "your company";
        dispatch(runBatchAnalysis({ files, jdFile, token, companyName }));
    }

    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30 overflow-x-hidden">
            {/* Background */}
            <div className="fixed inset-0 z-0 pointer-events-none bg-black">
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
                <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-white opacity-20 blur-[100px]" />
            </div>

            {/* Navbar */}
            <header className="fixed top-6 inset-x-0 z-50 flex justify-center px-6">
                <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 flex items-center justify-between shadow-2xl w-full max-w-6xl">
                    <a href="/" className="flex items-center hover:opacity-80 transition cursor-pointer pl-2">
                        <img src="/logo.png" alt="InSightATS Logo" className="h-10 md:h-12 w-auto object-contain scale-[2] md:scale-[2.5] origin-left" />
                    </a>
                    <div className="flex items-center gap-3">
                        <button onClick={() => nav("/")} className="hidden sm:flex items-center gap-2 px-4 h-10 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 transition">
                            <Home className="size-4" /> Home
                        </button>
                        <button onClick={logout} className="flex items-center gap-2 px-4 h-10 rounded-full bg-white text-black hover:opacity-90 transition">
                            <LogOut className="size-4" /> Logout
                        </button>
                    </div>
                </div>
            </header>

            <main className="relative z-10 w-full max-w-7xl mx-auto px-6 pt-28 pb-10">
                <MotionDiv initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                    <h1 className="text-3xl md:text-5xl font-bold mb-2 bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-transparent">
                        Recruiter Shortlist
                    </h1>
                    <p className="text-neutral-400 text-sm md:text-base max-w-2xl">
                        AI-powered ranking for candidate screening. Identify the best matches instantly with explainable metrics.
                    </p>
                </MotionDiv>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:h-[750px]">
                    {/* Left Panel: Inputs */}
                    <MotionDiv initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="lg:col-span-4 h-full">
                        <div className="group relative overflow-hidden rounded-3xl border border-white/10 bg-neutral-900/50 p-6 hover:bg-neutral-900/80 transition duration-500 h-full">
                            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500" />

                            <div className="relative z-10 space-y-6 h-full flex flex-col justify-between">
                                {/* Resume Upload */}
                                <div>
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                                            <Upload className="size-4 text-blue-400" /> Candidates
                                        </div>
                                        <Badge variant="outline" className="text-[10px] uppercase tracking-widest bg-black/40">
                                            {files.length} selected
                                        </Badge>
                                    </div>
                                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-white/10 border-dashed rounded-2xl cursor-pointer bg-black/20 hover:bg-black/40 hover:border-white/30 transition-all duration-300 group overflow-hidden relative">
                                        <div className="flex flex-col items-center justify-center pt-5 pb-6 z-10">
                                            <Upload className="w-8 h-8 mb-3 text-neutral-500 group-hover:text-blue-400 transition-colors" />
                                            <p className="mb-2 text-sm text-neutral-400"><span className="font-semibold text-white">Upload Resumes</span></p>
                                        </div>
                                        <input
                                            type="file"
                                            multiple
                                            className="hidden"
                                            onChange={(e) => dispatch(setFiles(Array.from(e.target.files || [])))}
                                        />
                                    </label>
                                    {files.length > 0 && (
                                        <div className="mt-4 text-xs space-y-2 max-h-32 overflow-y-auto pr-2 custom-scrollbar">
                                            {files.map((f) => (
                                                <div key={f.name} className="flex items-center gap-2 p-2.5 rounded-xl bg-white/5 border border-white/5 shadow-sm">
                                                    <FileText className="size-4 text-blue-400 shrink-0" />
                                                    <span className="truncate text-neutral-300 font-medium">{f.name}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* JD Upload */}
                                <div>
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                                            <FileText className="size-4 text-purple-400" /> Job Description
                                        </div>
                                        {jdFile && <Badge variant="success" className="text-[10px] uppercase">Ready</Badge>}
                                    </div>
                                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-white/10 border-dashed rounded-2xl cursor-pointer bg-black/20 hover:bg-black/40 hover:border-white/30 transition-all duration-300 group overflow-hidden relative">
                                        <div className="flex flex-col items-center justify-center pt-5 pb-6 z-10">
                                            <Upload className="w-8 h-8 mb-3 text-neutral-500 group-hover:text-purple-400 transition-colors" />
                                            <p className="mb-2 text-sm text-neutral-400"><span className="font-semibold text-white">Upload JD</span></p>
                                        </div>
                                        <input
                                            type="file"
                                            accept=".txt,.pdf,.docx"
                                            className="hidden"
                                            onChange={(e) => dispatch(setJdFile(e.target.files?.[0] || null))}
                                        />
                                    </label>
                                    {jdFile && (
                                        <div className="mt-4">
                                            <div className="flex items-center gap-2 p-2.5 rounded-xl bg-purple-500/10 border border-purple-500/20 shadow-sm">
                                                <FileText className="size-4 text-purple-400 shrink-0" />
                                                <span className="truncate font-medium text-purple-100 text-xs">{jdFile.name}</span>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Error */}
                                <AnimatePresence>
                                    {error && (
                                        <MotionDiv initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                                            <div className="bg-red-500/10 border border-red-500/20 text-red-200 px-4 py-3 rounded-xl text-sm flex items-center gap-2">
                                                <AlertTriangle className="size-4 shrink-0" /> {error}
                                            </div>
                                        </MotionDiv>
                                    )}
                                </AnimatePresence>

                                {/* CTA */}
                                <button
                                    disabled={loading || !files.length || !jdFile}
                                    onClick={analyze}
                                    className={`w-full h-14 rounded-full font-bold text-base shadow-lg flex items-center justify-center gap-2 transition-all duration-300 ${
                                        loading || !files.length || !jdFile
                                            ? "bg-neutral-800 text-neutral-500 cursor-not-allowed border border-white/5"
                                            : "bg-white text-black hover:scale-[1.02] shadow-[0_0_30px_-5px_rgba(255,255,255,0.3)]"
                                    }`}
                                >
                                    {loading ? (
                                        <div className="flex items-center gap-2">
                                            <div className="size-4 border-2 border-neutral-400 border-t-transparent rounded-full animate-spin" />
                                            <span>Processing...</span>
                                        </div>
                                    ) : (
                                        <>
                                            <span>Generate Shortlist</span>
                                            <ChevronRight className="size-4" />
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    </MotionDiv>

                    {/* Right Panel: Results */}
                    <MotionDiv
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="lg:col-span-8 group relative overflow-hidden rounded-3xl border border-white/10 bg-neutral-900/40 p-6 backdrop-blur-sm h-full"
                    >
                        <div className="absolute inset-0 bg-gradient-to-bl from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500" />
                        <div className="relative z-10 h-full flex flex-col">
                            <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/5">
                                <div className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                                    <Trophy className="size-5 text-yellow-500" /> Ranking
                                </div>
                                <div className="text-xs text-neutral-400 uppercase tracking-widest font-semibold bg-black/30 px-3 py-1 rounded-full border border-white/5">
                                    {data?.total ? `${data.total} Scored` : "Awaiting Input"}
                                </div>
                            </div>

                            {!data && !loading && (
                                <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50 my-12">
                                    <Trophy className="size-16 text-neutral-600 mb-4" />
                                    <p className="text-neutral-300 font-medium">No candidates analyzed yet</p>
                                    <p className="text-sm text-neutral-500 mt-1 max-w-xs">Upload resumes and a job description to generate your ranked shortlist.</p>
                                </div>
                            )}

                            {loading && (
                                <div className="flex-1 flex flex-col items-center justify-center my-12">
                                    <div className="size-12 border-4 border-white/10 border-t-white rounded-full animate-spin mb-4" />
                                    <p className="text-white font-medium animate-pulse">Running AI Analysis Pipeline...</p>
                                </div>
                            )}

                            {data?.results?.length > 0 && (
                                <MotionDiv
                                    variants={containerVariants}
                                    initial="hidden"
                                    animate="show"
                                    className="flex-1 min-h-0 overflow-y-auto pr-2 custom-scrollbar space-y-4 pb-4"
                                >
                                    {data.results.map((r) => {
                                        const an = r.analysis;
                                        return (
                                            <MotionDiv
                                                key={`${r.rank}-${r.filename}`}
                                                variants={itemVariants}
                                                className={`rounded-2xl border p-5 backdrop-blur-md transition-all duration-300 hover:scale-[1.01] ${getRankStyle(r.rank)}`}
                                            >
                                                <div className="flex flex-col md:flex-row md:items-start justify-between gap-5">
                                                    <div className="flex-1 min-w-0">
                                                        <div className="flex items-center gap-3 mb-2">
                                                            <div className="flex items-center justify-center size-10 rounded-full bg-black/40 border border-white/10 shadow-inner">
                                                                {getRankIcon(r.rank)}
                                                            </div>
                                                            <div className="min-w-0">
                                                                <h3 className="text-lg font-bold text-white truncate leading-tight">
                                                                    {r.filename.replace(/\.[^/.]+$/, "")}
                                                                </h3>
                                                                <div className="flex items-center gap-2 mt-1">
                                                                    <Badge variant="secondary" className="text-[10px] px-2 py-0 h-5 bg-white/10 border-white/5 hover:bg-white/20">
                                                                        {an?.fit_result?.verdict || "UNKNOWN FIT"}
                                                                    </Badge>
                                                                    {an?.missing_skills?.length > 0 ? (
                                                                        <span className="text-[10px] text-red-400 font-medium">Missing {an.missing_skills.length} skills</span>
                                                                    ) : (
                                                                        <span className="text-[10px] text-green-400 font-medium">All critical skills met</span>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        </div>

                                                        {an && (
                                                            <div className="grid grid-cols-5 gap-2 mt-4 max-w-xl">
                                                                {[
                                                                    { label: "AI Fit",  value: Math.round((an.fit_result?.fit_score || 0) * 100), color: "bg-pink-500" },
                                                                    { label: "Skills",  value: an.skill_score,      color: "bg-blue-500" },
                                                                    { label: "Exp",     value: an.experience_score, color: "bg-purple-500" },
                                                                    { label: "Proj",    value: an.project_score,    color: "bg-emerald-500" },
                                                                    { label: "Struct",  value: an.structure_score,  color: "bg-amber-500" },
                                                                ].map(({ label, value, color }) => (
                                                                    <div key={label} className="space-y-1.5">
                                                                        <div className="flex justify-between text-[10px] font-medium text-neutral-400 uppercase">
                                                                            <span>{label}</span><span className="text-white">{value}</span>
                                                                        </div>
                                                                        <Progress value={value} className="h-1 bg-white/10" indicatorClassName={color} />
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}

                                                        {an?.feedback?.length > 0 && (
                                                            <div className="mt-5 space-y-2 bg-black/20 p-3.5 rounded-xl border border-white/5">
                                                                <h4 className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider mb-2">Key Insights</h4>
                                                                {an.feedback.slice(0, 3).map((line, i) => (
                                                                    <div key={i} className="flex items-start gap-2 text-xs text-neutral-300">
                                                                        {getFeedbackIcon(line)}
                                                                        <span className="leading-snug" dangerouslySetInnerHTML={{ __html: cleanFeedbackText(line) }} />
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>

                                                    {/* Score Donut */}
                                                    <div className="shrink-0 flex flex-col items-center justify-center p-4 bg-black/40 rounded-2xl border border-white/10 min-w-[100px] shadow-inner">
                                                        <div className="relative flex items-center justify-center w-16 h-16 mb-1">
                                                            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                                                                <path className="text-white/10" strokeWidth="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                                                <motion.path
                                                                    initial={{ strokeDasharray: "0, 100" }}
                                                                    animate={{ strokeDasharray: `${r.score}, 100` }}
                                                                    transition={{ duration: 1.5, ease: "easeOut" }}
                                                                    className={r.score >= 80 ? "text-green-500" : r.score >= 60 ? "text-yellow-500" : "text-red-500"}
                                                                    strokeWidth="3"
                                                                    strokeLinecap="round"
                                                                    stroke="currentColor"
                                                                    fill="none"
                                                                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                                                />
                                                            </svg>
                                                            <div className="absolute inset-0 flex items-center justify-center">
                                                                <span className="text-xl font-black text-white">{r.score}</span>
                                                            </div>
                                                        </div>
                                                        <div className="text-[9px] uppercase tracking-widest text-neutral-400 font-bold">Match</div>
                                                    </div>
                                                </div>
                                            </MotionDiv>
                                        );
                                    })}
                                </MotionDiv>
                            )}
                        </div>
                    </MotionDiv>
                </div>
            </main>
        </div>
    );
}
