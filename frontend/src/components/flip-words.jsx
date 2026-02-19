"use client";
import React, { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { twMerge } from "tailwind-merge";

export const FlipWords = ({ words, duration = 3000, className }) => {
    const [currentWord, setCurrentWord] = useState(words[0]);
    const [isAnimating, setIsAnimating] = useState(false);

    // Find the longest word to reserve space
    const longestWord = words.reduce((a, b) => (a.length > b.length ? a : b));

    const startAnimation = useCallback(() => {
        const word = words[words.indexOf(currentWord) + 1] || words[0];
        setCurrentWord(word);
        setIsAnimating(true);
    }, [currentWord, words]);

    useEffect(() => {
        if (!isAnimating)
            setTimeout(() => {
                startAnimation();
            }, duration);
    }, [isAnimating, duration, startAnimation]);

    return (
        <span className="inline-grid grid-cols-1 grid-rows-1 items-baseline overflow-visible">
            {/* Ghost element: Hidden but maintains the width and height of the container */}
            <span className="invisible select-none pointer-events-none px-1 col-start-1 row-start-1" aria-hidden="true">
                {longestWord}
            </span>

            <AnimatePresence
                onExitComplete={() => {
                    setIsAnimating(false);
                }}
            >
                <motion.span
                    key={currentWord}
                    initial={{
                        opacity: 0,
                        y: 15,
                        filter: "blur(10px)",
                    }}
                    animate={{
                        opacity: 1,
                        y: 0,
                        filter: "blur(0px)",
                    }}
                    exit={{
                        opacity: 0,
                        y: -15,
                        filter: "blur(10px)",
                    }}
                    transition={{
                        type: "spring",
                        stiffness: 200,
                        damping: 20,
                    }}
                    className={twMerge("z-10 inline-block text-center whitespace-nowrap px-1 col-start-1 row-start-1", className)}
                >
                    {currentWord.split(" ").map((word, wordIndex) => (
                        <span key={word + wordIndex} className="inline-block">
                            {word.split("").map((letter, letterIndex) => (
                                <motion.span
                                    key={word + letterIndex}
                                    initial={{ opacity: 0, y: 10, filter: "blur(8px)" }}
                                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                                    transition={{
                                        delay: wordIndex * 0.05 + letterIndex * 0.02,
                                        duration: 0.2,
                                    }}
                                    className="inline-block"
                                >
                                    {letter}
                                </motion.span>
                            ))}
                            <span className="inline-block">&nbsp;</span>
                        </span>
                    ))}
                </motion.span>
            </AnimatePresence>
        </span>
    );
};