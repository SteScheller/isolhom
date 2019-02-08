#pragma once

namespace util
{
    class ProgressBar
    {
        public:
        ProgressBar();
        ProgressBar(unsigned int width, unsigned int iterations);

        /**
         * \brief Outputs the current state of progress as bar
         *
         * Outputs the current state of progress as bar and returns the
         * cursor to the beginning of the line such that subsequent calls
         * overwrite the last bar. To prevent that just print a newline
         * inbetween calls to this function.
         */
        void print();

        /** \brief Iterates the internal progress meter
         */
        ProgressBar& operator++();
        ProgressBar operator++(int dummy);

        private:
        unsigned int m_width;
        double m_incrementSize;
        double m_progress;
    };

}
