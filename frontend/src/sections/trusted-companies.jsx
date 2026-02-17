import SectionTitle from "../components/section-title";
import { motion } from "framer-motion";

export default function TrustedCompanies() {
    return (
        <section className="flex flex-col items-center">
            <SectionTitle title="Trusted companies" description="A visual collection of our most recent works - each piece crafted with intention, emotion, and style." />
            <motion.div className="relative max-w-5xl py-20 md:py-26 mt-18 md:w-full overflow-hidden mx-2 md:mx-auto border border-neutral-800 flex flex-col md:flex-row items-center justify-between bg-gradient-to-br from-[#171717]/10 to-[#0a0a0a]/20 rounded-3xl p-4 md:p-10 text-white"
                initial={{ y: 150, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ type: "spring", stiffness: 320, damping: 70, mass: 1 }}
            >
                <div className="absolute pointer-events-none top-10 -z-1 left-20 size-64 bg-gradient-to-br from-[#ffffff] to-[#e5e5e5]/60 blur-[180px] opacity-10"></div>
                <div className="absolute pointer-events-none bottom-10 -z-1 right-20 size-64 bg-gradient-to-br from-[#ffffff] to-[#e5e5e5]/60 blur-[180px] opacity-10"></div>
                <div className="flex flex-col items-center md:items-start max-md:text-center">
                    <a href="https://prebuiltui.com" className="group flex items-center gap-2 rounded-full text-sm p-1 pr-3 text-neutral-300 bg-neutral-800/50">
                        <span className="bg-white text-black text-xs px-3.5 py-1 rounded-full">
                            NEW
                        </span>
                        <p className="flex items-center gap-1">
                            <span>Try 30 days free trial option </span>
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-chevron-right-icon lucide-chevron-right group-hover:translate-x-0.5 transition duration-300"><path d="m9 18 6-6-6-6" /></svg>
                        </p>
                    </a>
                    <h1 className="text-3xl font-medium max-w-xl mt-5 bg-gradient-to-r from-white to-[#a3a3a3] text-transparent bg-clip-text">Trusted by Job Seekers.</h1>
                    <p className="text-base text-slate-400 max-w-lg mt-4">
                        Our AI Resume Analyzer has helped thousands of candidates land interviews at top companies.
                    </p>
                    <button className="flex items-center gap-1 text-sm px-6 py-2.5 border border-neutral-700 hover:bg-neutral-800 active:scale-95 transition rounded-full mt-6">
                        Read more
                        <svg width="13" height="10" viewBox="0 0 13 10" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12.4243 5.42426C12.6586 5.18995 12.6586 4.81005 12.4243 4.57574L8.60589 0.757359C8.37157 0.523045 7.99167 0.523045 7.75736 0.757359C7.52304 0.991674 7.52304 1.37157 7.75736 1.60589L11.1515 5L7.75736 8.39411C7.52304 8.62843 7.52304 9.00833 7.75736 9.24264C7.99167 9.47696 8.37157 9.47696 8.60589 9.24264L12.4243 5.42426ZM0 5L0 5.6L12 5.6V5V4.4L0 4.4L0 5Z" fill="white" />
                        </svg>
                    </button>
                </div>
                <div className="md:-mr-16 max-md:mt-10">
                    <img className="max-w-xs md:max-w-sm" src="https://raw.githubusercontent.com/prebuiltui/prebuiltui/main/assets/trusted-brand/image-fc6e.png" alt="" />
                </div>
            </motion.div>
        </section>
    );
}