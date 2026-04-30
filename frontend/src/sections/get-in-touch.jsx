import { useState, useRef } from "react";
import emailjs from "@emailjs/browser";
import SectionTitle from "../components/section-title";
import { ArrowUpRight, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { motion } from "framer-motion";

const SERVICE_ID  = import.meta.env.VITE_EMAILJS_SERVICE_ID;
const TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID;
const PUBLIC_KEY  = import.meta.env.VITE_EMAILJS_PUBLIC_KEY;

const MotionDiv    = motion.div;
const MotionButton = motion.button;

export default function GetInTouch() {
    const formRef = useRef(null);

    const [status, setStatus] = useState("idle"); // idle | loading | success | error
    const [errMsg, setErrMsg] = useState("");

    async function handleSubmit(e) {
        e.preventDefault();
        if (status === "loading") return;

        const form = formRef.current;
        // Access named inputs via form.elements to avoid conflicts with native form properties
        const nameVal    = form.elements["from_name"].value.trim();
        const emailVal   = form.elements["email"].value.trim();
        const messageVal = form.elements["message"].value.trim();

        if (!nameVal || !emailVal || !messageVal) {
            setErrMsg("Please fill in all fields before submitting.");
            setStatus("error");
            return;
        }

        // Graceful fallback if EmailJS keys are still placeholders (dev environment)
        if (!SERVICE_ID || SERVICE_ID === "your_service_id") {
            setErrMsg("EmailJS is not configured yet. Add VITE_EMAILJS_SERVICE_ID, VITE_EMAILJS_TEMPLATE_ID, and VITE_EMAILJS_PUBLIC_KEY to your .env.local file.");
            setStatus("error");
            return;
        }

        setStatus("loading");
        setErrMsg("");

        try {
            await emailjs.sendForm(SERVICE_ID, TEMPLATE_ID, form, { publicKey: PUBLIC_KEY });
            setStatus("success");
            form.reset();
        } catch (err) {
            setErrMsg(err?.text || "Failed to send your message. Please try again.");
            setStatus("error");
        }
    }

    const inputClass =
        "w-full mt-2 p-3 bg-transparent outline-none border border-slate-700 rounded-lg focus:ring-1 transition focus:ring-primary placeholder:text-slate-600 text-white disabled:opacity-50";

    return (
        <section className="flex flex-col items-center" id="contact">
            <SectionTitle
                title="Get in touch"
                description="Have questions about our AI-powered resume analysis or want to learn more about our intelligent recruitment solutions? We'd love to hear from you."
            />

            <form
                ref={formRef}
                onSubmit={handleSubmit}
                className="grid sm:grid-cols-2 gap-3 sm:gap-5 max-w-3xl mx-auto text-slate-400 mt-16 w-full"
            >
                <MotionDiv
                    initial={{ y: 150, opacity: 0 }}
                    whileInView={{ y: 0, opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ type: "spring", stiffness: 320, damping: 70, mass: 1 }}
                >
                    <label className="font-medium text-slate-200">Your name</label>
                    <input
                        name="from_name"
                        type="text"
                        placeholder="Enter your name"
                        className={inputClass}
                        disabled={status === "loading"}
                    />
                </MotionDiv>

                <MotionDiv
                    initial={{ y: 150, opacity: 0 }}
                    whileInView={{ y: 0, opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ type: "spring", stiffness: 280, damping: 70, mass: 1 }}
                >
                    <label className="font-medium text-slate-200">Email address</label>
                    <input
                        name="email"
                        type="email"
                        placeholder="Enter your email"
                        className={inputClass}
                        disabled={status === "loading"}
                    />
                </MotionDiv>

                <MotionDiv
                    className="sm:col-span-2"
                    initial={{ y: 150, opacity: 0 }}
                    whileInView={{ y: 0, opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ type: "spring", stiffness: 240, damping: 70, mass: 1 }}
                >
                    <label className="font-medium text-slate-200">Message</label>
                    <textarea
                        name="message"
                        rows={7}
                        placeholder="Enter your message"
                        className={`resize-none ${inputClass}`}
                        disabled={status === "loading"}
                    />
                </MotionDiv>

                {/* Status feedback */}
                {status === "success" && (
                    <MotionDiv
                        className="sm:col-span-2 flex items-center gap-3 bg-green-500/10 border border-green-500/20 rounded-xl px-4 py-3 text-green-300 text-sm"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <CheckCircle className="size-4 shrink-0" />
                        Your message has been sent successfully. We'll be in touch soon!
                    </MotionDiv>
                )}

                {status === "error" && errMsg && (
                    <MotionDiv
                        className="sm:col-span-2 flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 text-red-300 text-sm"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        <AlertCircle className="size-4 shrink-0" />
                        {errMsg}
                    </MotionDiv>
                )}

                <MotionButton
                    type="submit"
                    disabled={status === "loading"}
                    className={`w-max flex items-center gap-2 px-8 py-3 rounded-full font-medium transition-all duration-200 active:scale-95 ${
                        status === "loading"
                            ? "bg-primary/60 text-black/60 cursor-not-allowed"
                            : "bg-primary hover:bg-white text-black"
                    }`}
                    initial={{ y: 150, opacity: 0 }}
                    whileInView={{ y: 0, opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ type: "spring", stiffness: 280, damping: 70, mass: 1 }}
                >
                    {status === "loading" ? (
                        <>
                            <Loader2 className="size-4 animate-spin" />
                            Sending…
                        </>
                    ) : (
                        <>
                            Send message
                            <ArrowUpRight className="size-4.5" />
                        </>
                    )}
                </MotionButton>
            </form>
        </section>
    );
}