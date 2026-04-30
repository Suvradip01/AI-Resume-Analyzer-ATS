import SectionTitle from "../components/section-title";
import { memo } from "react";
import {
    FileScan,
    Network,
    GitMerge,
    Gauge,
    Sparkles,
    Target,
} from "lucide-react";

const FeatureCard = memo(({ feature }) => (
    <div className="p-6 rounded-xl mx-4 bg-slate-50 border border-slate-200 shadow-sm transition-all duration-300 w-80 shrink-0 group">
        <div className="flex gap-4 items-center">
            {/* Elegant Minimalist Dark Icon Block with dynamic colors */}
            <div className={`size-12 rounded-xl border flex items-center justify-center shrink-0 shadow-sm group-hover:scale-105 group-hover:-translate-y-0.5 transition-all duration-300 relative overflow-hidden ${feature.bgColor} ${feature.borderColor}`}>
                <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <feature.icon className={`size-5 stroke-[1.5] relative z-10 ${feature.iconColor}`} />
            </div>
            <div className="flex flex-col">
                <h3 className="text-base font-bold text-slate-900">{feature.title}</h3>
                <div className="flex items-center gap-1 mt-0.5">
                    <span className="text-[10px] uppercase tracking-wider text-slate-500 font-bold whitespace-nowrap">AI Pipeline</span>
                    <svg className="fill-slate-500 shrink-0" width="10" height="10" viewBox="0 0 12 12">
                        <path d="M4.555.72a4 4 0 0 1-.297.24c-.179.12-.38.202-.59.244a4 4 0 0 1-.38.041c-.48.039-.721.058-.922.129a1.63 1.63 0 0 0-.992.992c-.071.2-.09.441-.129.922a4 4 0 0 1-.041.38 1.6 1.6 0 0 1-.245.59 3 3 0 0 1-.239.297c-.313.368-.47.551-.56.743-.213.444-.213.96 0 1.404.09.192.247.375.56.743.125.146.187.219.24.297.12.179.202.38.244.59.018.093.026.189.041.38.039.48.058.721.129.922.163.464.528.829.992.992.2.071.441.09.922.129.191.015.287.023.38.041.21.042.411.125.59.245.078.052.151.114.297.239.368.313.551.47.743.56.444.213.96.213 1.404 0 .192-.09.375-.247.743-.56.146-.125.219-.187.297-.24.179-.12.38-.202.59-.244a4 4 0 0 1 .38-.041c.48-.039.721-.058.922-.129.464-.163.829-.528.992-.992.071-.2.09-.441.129-.922a4 4 0 0 1 .041-.38c.042-.21.125-.411.245-.59.052-.078.114-.151.239-.297.313-.368.47-.551.56-.743.213-.444.213-.96 0-1.404-.09-.192-.247-.375-.56-.743a4 4 0 0 1-.24-.297 1.6 1.6 0 0 1-.244-.59 3 3 0 0 1-.041-.38c-.039-.48-.058-.721-.129-.922a1.63 1.63 0 0 0-.992-.992c-.2-.071-.441-.09-.922-.129a4 4 0 0 1-.38-.041 1.6 1.6 0 0 1-.59-.245A3 3 0 0 1 7.445.72C7.077.407 6.894.25 6.702.16a1.63 1.63 0 0 0-1.404 0c-.192.09-.375.247-.743.56m4.07 3.998a.488.488 0 0 0-.691-.69l-2.91 2.91-.958-.957a.488.488 0 0 0-.69.69l1.302 1.302c.19.191.5.191.69 0z" />
                    </svg>
                </div>
            </div>
        </div>
        <p className="mt-4 text-sm leading-relaxed text-slate-600 group-hover:text-slate-800 transition-colors font-medium">
            {feature.description}
        </p>
    </div>
));

export default function AboutOurApps() {
    const featuresData = [
        {
            title: "Smart Resume Parsing",
            description: "Effortlessly processes any format. Our intelligent parser accurately extracts your experience and skills without missing a beat.",
            icon: FileScan,
            iconColor: "text-emerald-500",
            bgColor: "bg-emerald-500/10",
            borderColor: "border-emerald-500/20",
        },
        {
            title: "Deep Context Extraction",
            description: "Goes beyond simple keyword matching. Our AI understands your actual roles and uncovers capabilities that traditional systems overlook.",
            icon: Network,
            iconColor: "text-orange-500",
            bgColor: "bg-orange-500/10",
            borderColor: "border-orange-500/20",
        },
        {
            title: "True Semantic Matching",
            description: "Matches candidates to jobs based on real meaning and potential, not just buzzwords. We connect the right talent with the right roles.",
            icon: GitMerge,
            iconColor: "text-yellow-500",
            bgColor: "bg-yellow-500/10",
            borderColor: "border-yellow-500/20",
        },
        {
            title: "Project Impact Analysis",
            description: "Automatically evaluates the real-world complexity of your work, giving fair credit to candidates with hands-on, production-grade experience.",
            icon: Gauge,
            iconColor: "text-emerald-500",
            bgColor: "bg-emerald-500/10",
            borderColor: "border-emerald-500/20",
        },
        {
            title: "Transparent AI Decisions",
            description: "No black boxes here. Get clear, understandable reasons behind every score, ensuring absolute fairness in the recruitment process.",
            icon: Sparkles,
            iconColor: "text-orange-500",
            bgColor: "bg-orange-500/10",
            borderColor: "border-orange-500/20",
        },
        {
            title: "Comprehensive Scoring",
            description: "Receive a holistic evaluation across four key dimensions, giving recruiters a confident shortlist and candidates actionable feedback.",
            icon: Target,
            iconColor: "text-yellow-500",
            bgColor: "bg-yellow-500/10",
            borderColor: "border-yellow-500/20",
        },
    ];

    return (
        <section className="flex flex-col items-center py-20 overflow-hidden" id="about">
            <style>{`
                @keyframes marqueeScroll {
                    0% { transform: translateX(0%); }
                    100% { transform: translateX(-50%); }
                }

                .marquee-inner {
                    animation: marqueeScroll 35s linear infinite;
                    will-change: transform;
                }

                .marquee-reverse {
                    animation-direction: reverse;
                }
                
                .marquee-inner:hover {
                    animation-play-state: paused;
                }
            `}</style>

            <div className="px-6 md:px-16 lg:px-24 xl:px-32 w-full flex flex-col items-center text-center">
                <SectionTitle
                    title="How it works"
                    description="A transparent, multi-stage transformer pipeline that explains every decision it makes."
                />
            </div>

            <div className="w-full mt-16 relative">
                {/* Gradient Masks */}
                <div className="absolute left-0 top-0 h-full w-24 md:w-60 z-10 pointer-events-none bg-gradient-to-r from-black to-transparent"></div>
                <div className="absolute right-0 top-0 h-full w-24 md:w-60 z-10 pointer-events-none bg-gradient-to-l from-black to-transparent"></div>

                {/* First Row: Normal Scroll */}
                <div className="marquee-row w-full overflow-hidden relative pb-10">
                    <div className="marquee-inner flex transform-gpu min-w-[200%]">
                        {[...featuresData, ...featuresData].map((feature, index) => (
                            <FeatureCard key={`row1-${index}`} feature={feature} />
                        ))}
                    </div>
                </div>

                {/* Second Row: Reverse Scroll */}
                <div className="marquee-row w-full overflow-hidden relative">
                    <div className="marquee-inner marquee-reverse flex transform-gpu min-w-[200%]">
                        {[...featuresData, ...featuresData].map((feature, index) => (
                            <FeatureCard key={`row2-${index}`} feature={feature} />
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}