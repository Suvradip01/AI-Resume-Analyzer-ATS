import { ArrowRight, User, Building2 } from "lucide-react";
import { motion } from "framer-motion";
import TiltedImage from "../components/tilt-image";
import { Link } from "react-router-dom";
import { FlipWords } from "../components/flip-words";

export default function HeroSection() {
    const MotionSvg = motion.svg;
    const MotionDiv = motion.div;
    const MotionH1 = motion.h1;
    const MotionP = motion.p;

    const words = ["Dream job", "Next Step", "True Path"];

    return (
        <section className="flex flex-col items-center -mt-18">
            <MotionSvg className="absolute -z-10 w-full -mt-40 md:mt-0" width="1440" height="676" viewBox="0 0 1440 676" fill="none" xmlns="http://www.w3.org/2000/svg"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
            >
                <rect x="-92" y="-948" width="1624" height="1624" rx="812" fill="url(#a)" />
                <defs>
                    <radialGradient id="a" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="rotate(90 428 292)scale(812)">
                        <stop offset=".63" stopColor="#ffffff" stopOpacity="0" />
                        <stop offset="1" stopColor="#ffffff" />
                    </radialGradient>
                </defs>
            </MotionSvg>
            <MotionDiv className="flex items-center mt-48 gap-2 border border-slate-600 text-gray-50 rounded-full px-4 py-2"
                initial={{ y: -20, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2, type: "spring", stiffness: 320, damping: 70, mass: 1 }}
            >
                <div className="size-2.5 bg-green-500 rounded-full animate-pulse"></div>
                <span>AI Powered Resume Analysis</span>
            </MotionDiv>
            <MotionH1 className="text-center text-4xl md:text-6xl lg:text-7xl mt-4 font-bold tracking-tight leading-tight md:leading-[1.1] max-w-4xl mx-auto"
                initial={{ y: 50, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 240, damping: 70, mass: 1 }}
            >
                <div className="flex flex-wrap justify-center items-center gap-x-3">
                    <span>Get your</span>
                    <span className="text-primary-mid">
                        <FlipWords words={words} />
                    </span>
                </div>
                <div className="text-white/60 text-3xl md:text-5xl mt-3 font-medium">with AI</div>
            </MotionH1>
            <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="text-neutral-400 text-base md:text-lg lg:text-xl max-w-4xl mx-auto mb-4 leading-relaxed font-medium text-center"
            >
                Transparent 4D scoring and explainable reasoning to help you stand out.
            </motion.p>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1, delay: 0.4 }}
                className="flex items-center justify-center gap-3 text-[10px] font-bold uppercase tracking-[0.2em] text-white/30 mb-10"
            >
                <span>BERT</span>
                <span className="size-1 rounded-full bg-white/20" />
                <span>RoBERTa</span>
                <span className="size-1 rounded-full bg-white/20" />
                <span>DistilBERT</span>
                <span className="size-1 rounded-full bg-white/20" />
                <span>SHAP</span>
            </motion.div>
            <MotionDiv className="flex flex-col sm:flex-row items-center gap-4 mt-8"
                initial={{ y: 50, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 320, damping: 70, mass: 1 }}
            >
                <a href="/dashboard" className="flex items-center gap-2 bg-primary hover:bg-white transition text-black active:scale-95 font-medium rounded-xl px-7 h-12 shadow-[0_0_20px_rgba(255,255,255,0.3)]">
                    <User className="size-5" />
                    For Candidates
                    <ArrowRight className="size-4 ml-1" />
                </a>
                <a href="/recruiter" className="flex items-center gap-2 bg-neutral-900 border border-white/10 hover:bg-neutral-800 hover:border-white/30 transition text-white active:scale-95 font-medium rounded-xl px-7 h-12">
                    <Building2 className="size-5 text-neutral-400" />
                    For Recruiters
                    <ArrowRight className="size-4 ml-1 text-neutral-400" />
                </a>
            </MotionDiv>
            <TiltedImage />
        </section>
    );
}
