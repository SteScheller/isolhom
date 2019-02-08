#include <iostream>

#include "progbar.hpp"


//-----------------------------------------------------------------------------
// class implementations
//-----------------------------------------------------------------------------
util::ProgressBar::ProgressBar() :
    m_width(50),
    m_incrementSize(0.01),
    m_progress(0.0)
{
}

util::ProgressBar::ProgressBar(
        unsigned int width, unsigned int iterations) :
    m_width(width),
    m_incrementSize(1.0 / static_cast<double>(iterations)),
    m_progress(0.0)
{
}

void util::ProgressBar::print()
{
    if (m_progress < 1.0)
    {
        std::cout << "[";
        unsigned int pos = m_width * m_progress;
        for (unsigned int i = 0; i < m_width; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(m_progress * 100.0) << " %\r";
    }
    else
    {
        std::cout << "[";
        for (unsigned int i = 0; i < m_width; ++i)
            std::cout << "=";
        std::cout << "] 100 % Done!" << std::endl;
    }
}

util::ProgressBar& util::ProgressBar::operator++()
{
    m_progress += m_incrementSize;

    return *this;
}

util::ProgressBar util::ProgressBar::operator++(
        __attribute__((unused)) int dummy)
{
    util::ProgressBar old = *this;
    m_progress += m_incrementSize;

    return old;
}

