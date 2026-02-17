import { UserButton, useUser } from "@clerk/clerk-react";
import { useState } from "react";
import { Upload, FileText, CheckCircle, AlertCircle, Sparkles, ChevronRight, X, ChevronLeft } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ResultPieChart from "../components/ResultPieChart";
import ResultRadarChart from "../components/ResultRadarChart";

export default function Dashboard() {
    const { user } = useUser();
    const [file, setFile] = useState(null);
    const [jobDescription, setJobDescription] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected) {
            if (selected.type === "application/pdf" || selected.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" || selected.type === "text/plain") {
                setFile(selected);
                setError("");
            } else {
                setError("Please upload a PDF, DOCX, or TXT file.");
            }
        }
    };

    const handleAnalyze = async () => {
        if (!file || !jobDescription) {
            setError("Please provide both a resume and a job description.");
            return;
        }

        setLoading(true);
        setError("");
        setResult(null);

        const formData = new FormData();
        formData.append("resume_file", file);

        const jobData = JSON.stringify({
            title: "Target Role",
            description: jobDescription
        });

        formData.append("job_description", jobData);

        try {
            const response = await fetch("http://127.0.0.1:8000/api/v1/resume/analyze", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Analysis failed");
            }

            const data = await response.json();
            setResult(data);
            // Smooth scroll to results
            setTimeout(() => {
                document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
            }, 100);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/30 overflow-x-hidden">
            {/* Background Gradients & Grid */}
            <div className="fixed inset-0 z-0 pointer-events-none bg-black">
                {/* Grid Pattern */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
                {/* Radial gradient container */}
                <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-white opacity-20 blur-[100px]"></div>
            </div>

            {/* Header / Floating Navbar */}
            <header className="fixed top-6 inset-x-0 z-50 flex justify-center px-6">
                <div className="bg-black/40 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 flex items-center justify-between shadow-2xl w-full max-w-6xl">

                    <a href="/" className="flex items-center gap-3 hover:opacity-80 transition cursor-pointer">
                        <div className="size-8 bg-white rounded-lg flex items-center justify-center shadow-lg shadow-white/20">
                            <Sparkles className="text-black size-5" />
                        </div>
                        <span className="font-bold text-lg tracking-tight text-white">InSight ATS</span>
                    </a>

                    <div className="flex items-center gap-6">
                        {/* Status Indicator */}
                        <div className="hidden md:flex items-center gap-3 px-4 py-2 bg-white/5 rounded-full border border-white/5 backdrop-blur-sm">
                            <span className="text-xs font-medium text-slate-400">System Operational</span>
                            <div className="flex gap-1.5">
                                <span className="relative flex h-2 w-2">
                                    <span className="animate-[ping_1.5s_linear_infinite] absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                                </span>
                                <span className="relative flex h-2 w-2">
                                    <span className="animate-[ping_1.5s_linear_infinite] delay-500 absolute inline-flex h-full w-full rounded-full bg-orange-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-orange-500"></span>
                                </span>
                                <span className="relative flex h-2 w-2">
                                    <span className="animate-[ping_1.5s_linear_infinite] delay-1000 absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                                </span>
                            </div>
                        </div>

                        <UserButton afterSignOutUrl="/" appearance={{
                            elements: {
                                avatarBox: "size-9 border-2 border-white/10 ring-2 ring-white/5 hover:ring-white/20 transition duration-300"
                            }
                        }} />
                    </div>
                </div>
            </header>

            <main className="relative z-10 w-full max-w-6xl mx-auto px-6 pt-28 pb-10 flex flex-col items-center justify-center min-h-screen">

                {/* Intro - Compact */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-8"
                >
                    <h1 className="text-3xl md:text-5xl font-bold mb-3 bg-gradient-to-b from-white to-neutral-400 bg-clip-text text-transparent">
                        Optimize Your Resume
                    </h1>
                    <p className="text-neutral-400 text-sm md:text-base max-w-lg mx-auto">
                        Get instant, AI-powered compatibility scoring and feedback.
                    </p>
                </motion.div>

                {/* Main Content Grid */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1, duration: 0.4 }}
                    className="w-full grid grid-cols-1 lg:grid-cols-2 gap-6"
                >
                    {/* Left Column: Input Zone */}
                    <div className="flex flex-col gap-6">
                        {/* Resume Upload Card */}
                        <div className="group relative overflow-hidden rounded-3xl border border-white/10 bg-neutral-900/50 p-6 hover:bg-neutral-900/80 transition duration-500">
                            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500" />

                            <div className="flex items-center justify-between mb-4 relative z-10">
                                <label className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                                    Resume File
                                </label>
                                {file && <span className="text-xs font-medium text-black bg-white px-2 py-1 rounded-full">Uploaded</span>}
                            </div>

                            <div className="relative">
                                <input
                                    type="file"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                                />
                                <div className={`
                                    border border-dashed rounded-2xl h-40 transition-all duration-300 flex flex-col items-center justify-center gap-3
                                    ${file
                                        ? 'border-white/40 bg-white/5'
                                        : 'border-neutral-700 hover:border-white/50 hover:bg-white/5'
                                    }
                                `}>
                                    {file ? (
                                        <>
                                            <div className="size-10 bg-white rounded-xl flex items-center justify-center shadow-lg">
                                                <CheckCircle className="size-5 text-black" />
                                            </div>
                                            <div className="text-center">
                                                <p className="font-medium text-white text-sm truncate max-w-[200px]">{file.name}</p>
                                                <p className="text-xs text-neutral-400">Ready for analysis</p>
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <div className="size-10 rounded-xl bg-neutral-800 flex items-center justify-center text-neutral-400 group-hover:bg-white group-hover:text-black transition duration-300">
                                                <Upload className="size-5" />
                                            </div>
                                            <div className="text-center">
                                                <p className="text-sm font-medium text-neutral-300">Click to upload</p>
                                                <p className="text-[10px] text-neutral-500 uppercase tracking-widest mt-1">PDF • DOCX • TXT</p>
                                            </div>
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Job Desc & Action */}
                    <div className="flex flex-col gap-6 h-full">
                        {/* Job Description Card */}
                        <div className="group relative flex-1 overflow-hidden rounded-3xl border border-white/10 bg-neutral-900/50 p-6 hover:bg-neutral-900/80 transition duration-500">
                            <div className="absolute inset-0 bg-gradient-to-bl from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500" />

                            <div className="flex items-center justify-between mb-4 relative z-10">
                                <label className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                                    Job Description
                                </label>
                                <span className="text-[10px] text-neutral-500">{jobDescription.length} chars</span>
                            </div>

                            <textarea
                                value={jobDescription}
                                onChange={(e) => setJobDescription(e.target.value)}
                                placeholder="Paste the job description here..."
                                className="relative z-20 w-full h-[calc(100%-3rem)] bg-transparent border-none text-neutral-200 focus:ring-0 resize-none placeholder:text-neutral-600 text-sm leading-relaxed p-0 scrollbar-hide"
                            />
                        </div>
                    </div>
                </motion.div>

                {/* Floating Action Button - Positioned Centered overlapping the grid or below */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="w-full max-w-md mt-8"
                >
                    <AnimatePresence>
                        {error && (
                            <motion.div
                                initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                                animate={{ opacity: 1, height: "auto", marginBottom: 12 }}
                                exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="bg-red-500/10 border border-red-500/20 text-red-200 px-4 py-2 rounded-lg flex items-center gap-2 text-sm justify-center">
                                    <AlertCircle className="size-4 shrink-0" />
                                    <span>{error}</span>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    <button
                        onClick={handleAnalyze}
                        disabled={loading}
                        className={`
                            relative group w-full h-14 rounded-full font-bold text-base shadow-[0_0_40px_-10px_rgba(255,255,255,0.3)] flex items-center justify-center gap-2 overflow-hidden transition-all duration-300
                            ${loading
                                ? 'bg-neutral-900 text-neutral-500 cursor-not-allowed border border-white/5'
                                : 'bg-white text-black hover:scale-[1.02]'
                            }
                        `}
                    >
                        {loading ? (
                            <>
                                <div className="size-4 border-2 border-neutral-400 border-t-transparent rounded-full animate-spin" />
                                <span>Processing...</span>
                            </>
                        ) : (
                            <>
                                <span>Run Analysis</span>
                                <ChevronRight className="size-4 group-hover:translate-x-1 transition-transform" />
                            </>
                        )}
                    </button>
                </motion.div>

                {/* Value Props (Only show if no result yet) */}
                {!result && !loading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5 }}
                        className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16 w-full opacity-60 hover:opacity-100 transition duration-500"
                    >
                        <div className="text-center p-4">
                            <div className="bg-white/5 w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4 border border-white/10">
                                <FileText className="text-primary size-6" />
                            </div>
                            <h3 className="text-white font-semibold mb-1">ATS Friendly</h3>
                            <p className="text-sm text-slate-400">Checks if your resume parses correctly.</p>
                        </div>
                        <div className="text-center p-4">
                            <div className="bg-white/5 w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4 border border-white/10">
                                <Sparkles className="text-accent size-6" />
                            </div>
                            <h3 className="text-white font-semibold mb-1">Keyword Match</h3>
                            <p className="text-sm text-slate-400">Finds missing keywords from the job description.</p>
                        </div>
                        <div className="text-center p-4">
                            <div className="bg-white/5 w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-4 border border-white/10">
                                <CheckCircle className="text-emerald-400 size-6" />
                            </div>
                            <h3 className="text-white font-semibold mb-1">Score & Tips</h3>
                            <p className="text-sm text-slate-400">Get a 0-100 score and actionable feedback.</p>
                        </div>
                    </motion.div>
                )}

                {/* Results Section */}
                <AnimatePresence>
                    {result && (
                        <motion.div
                            id="results-section"
                            initial={{ opacity: 0, y: 40 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="w-full mt-10 space-y-8 pb-20"
                        >
                            <div className="flex items-center justify-center gap-4 py-8">
                                <div className="h-px bg-white/10 flex-1" />
                                <span className="text-primary font-semibold tracking-wider uppercase text-sm">Analysis Results</span>
                                <div className="h-px bg-white/10 flex-1" />
                            </div>

                            {/* Charts Section */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                {/* Overall Score - Pie Chart */}
                                <div className="bg-secondary/40 backdrop-blur-md border border-white/10 rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden">
                                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">Overall Match Score</h2>
                                    <ResultPieChart score={result.score} />
                                </div>

                                {/* XAI Analysis - Radar Chart */}
                                <div className="bg-secondary/40 backdrop-blur-md border border-white/10 rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden">
                                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">XAI Match Breakdown</h2>
                                    <ResultRadarChart data={[
                                        { subject: 'Skills', A: result.skill_score, fullMark: 100 },
                                        { subject: 'Experience', A: result.experience_score, fullMark: 100 },
                                        { subject: 'Projects', A: result.project_score, fullMark: 100 },
                                        { subject: 'Structure', A: result.structure_score, fullMark: 100 },
                                    ]} />
                                </div>
                            </div>

                            {/* Missing Skills */}
                            {result.missing_skills && result.missing_skills.length > 0 && (
                                <div className="bg-white/5 border border-white/10 rounded-3xl p-8">
                                    <h3 className="flex items-center gap-3 font-bold text-red-300 mb-6 text-lg">
                                        <AlertCircle className="size-5" />
                                        <span>Missing Keywords</span>
                                    </h3>
                                    <div className="flex flex-wrap gap-3">
                                        {result.missing_skills.map((skill, i) => (
                                            <span key={i} className="px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-200 text-sm font-medium rounded-xl">
                                                {skill}
                                            </span>
                                        ))}
                                    </div>
                                    <p className="mt-4 text-sm text-slate-400">
                                        tip: enhancing your resume with these keywords can significantly improve your ATS score.
                                    </p>
                                </div>
                            )}

                            {/* Detailed Feedback */}
                            <div className="space-y-4">
                                <h3 className="font-bold text-white text-xl pl-2">Detailed Insights</h3>
                                {result.feedback.map((item, idx) => (
                                    <div
                                        key={idx}
                                        className="bg-secondary/50 border border-white/5 rounded-2xl p-6 text-slate-300 text-base leading-relaxed"
                                    >
                                        <div dangerouslySetInnerHTML={{
                                            __html: item
                                                .replace(/\*\*(.*?)\*\*/g, '<strong class="text-primary-mid font-semibold">$1</strong>')
                                                .replace(/❌/g, '<span class="inline-flex size-6 items-center justify-center bg-red-500/20 rounded-full text-xs mr-3 align-middle">✕</span>')
                                                .replace(/✅/g, '<span class="inline-flex size-6 items-center justify-center bg-green-500/20 rounded-full text-xs mr-3 align-middle">✓</span>')
                                                .replace(/⚠️/g, '<span class="inline-flex size-6 items-center justify-center bg-amber-500/20 rounded-full text-xs mr-3 align-middle">!</span>')
                                                .replace(/💡/g, '<span class="inline-flex size-6 items-center justify-center bg-primary/20 rounded-full text-xs mr-3 align-middle">💡</span>')
                                        }} />
                                    </div>
                                ))}
                            </div>

                        </motion.div>
                    )}
                </AnimatePresence>
            </main>
        </div>
    );
}
