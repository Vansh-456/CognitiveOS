/**
 * Cognitive State Compilation (CSC) - Interactive Presentation Logic
 * Features:
 * - Keyboard navigation (Arrows, Space, F, O, P)
 * - Elapsed presenter timer & speaker notes drawer
 * - Slide Overview modal with instant thumbnail jump
 * - Lightbox modal for zooming figures
 * - Theme switcher (Dark / Light)
 * - Print / PDF export
 */

class SlideDeck {
  constructor() {
    this.slides = Array.from(document.querySelectorAll('.slide'));
    this.currentIndex = 0;
    this.totalSlides = this.slides.length;
    
    // UI Elements
    this.progressBar = document.getElementById('progressBar');
    this.slideCounter = document.getElementById('slideCounter');
    this.btnPrev = document.getElementById('btnPrev');
    this.btnNext = document.getElementById('btnNext');
    this.presenterDrawer = document.getElementById('presenterDrawer');
    this.presenterContent = document.getElementById('presenterContent');
    this.timerDisplay = document.getElementById('timerDisplay');
    this.overviewModal = document.getElementById('overviewModal');
    this.overviewGrid = document.getElementById('overviewGrid');
    this.lightboxModal = document.getElementById('lightboxModal');
    this.lightboxImg = document.getElementById('lightboxImg');
    
    // State
    this.timerSeconds = 0;
    this.timerInterval = null;
    this.isPresenterOpen = false;
    
    this.init();
  }

  init() {
    this.setupEvents();
    this.renderSlide(0);
    this.buildOverviewGrid();
    this.startTimer();
  }

  setupEvents() {
    // Navigation Buttons
    this.btnPrev.addEventListener('click', () => this.prevSlide());
    this.btnNext.addEventListener('click', () => this.nextSlide());
    
    // Header Buttons
    document.getElementById('btnOverview').addEventListener('click', () => this.toggleOverview());
    document.getElementById('btnCloseOverview').addEventListener('click', () => this.toggleOverview(false));
    document.getElementById('btnNotes').addEventListener('click', () => this.togglePresenterDrawer());
    document.getElementById('btnCloseNotes').addEventListener('click', () => this.togglePresenterDrawer(false));
    document.getElementById('btnFullscreen').addEventListener('click', () => this.toggleFullscreen());
    document.getElementById('btnTheme').addEventListener('click', () => this.toggleTheme());
    document.getElementById('btnPrint').addEventListener('click', () => window.print());

    // Lightbox Close
    this.lightboxModal.addEventListener('click', () => {
      this.lightboxModal.classList.remove('open');
    });

    // Zoomable images
    document.querySelectorAll('.fig-img').forEach(img => {
      img.addEventListener('click', (e) => {
        e.stopPropagation();
        this.lightboxImg.src = img.src;
        this.lightboxModal.classList.add('open');
      });
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Don't intercept if inside an input or modal is open (except ESC)
      if (e.key === 'Escape') {
        if (this.lightboxModal.classList.contains('open')) {
          this.lightboxModal.classList.remove('open');
          return;
        }
        if (this.overviewModal.classList.contains('open')) {
          this.toggleOverview(false);
          return;
        }
        if (this.isPresenterOpen) {
          this.togglePresenterDrawer(false);
          return;
        }
      }

      if (this.overviewModal.classList.contains('open')) return;

      switch(e.key) {
        case 'ArrowRight':
        case 'PageDown':
        case ' ': // Spacebar
          e.preventDefault();
          this.nextSlide();
          break;
        case 'ArrowLeft':
        case 'PageUp':
        case 'Backspace':
          e.preventDefault();
          this.prevSlide();
          break;
        case 'Home':
          e.preventDefault();
          this.goToSlide(0);
          break;
        case 'End':
          e.preventDefault();
          this.goToSlide(this.totalSlides - 1);
          break;
        case 'f':
        case 'F':
          this.toggleFullscreen();
          break;
        case 'o':
        case 'O':
          this.toggleOverview();
          break;
        case 'p':
        case 'P':
          this.togglePresenterDrawer();
          break;
      }
    });

    // Touch Swipes
    let touchStartX = 0;
    document.addEventListener('touchstart', (e) => {
      touchStartX = e.changedTouches[0].screenX;
    }, false);

    document.addEventListener('touchend', (e) => {
      let touchEndX = e.changedTouches[0].screenX;
      let diff = touchStartX - touchEndX;
      if (Math.abs(diff) > 50) {
        if (diff > 0) this.nextSlide();
        else this.prevSlide();
      }
    }, false);
  }

  renderSlide(index) {
    if (index < 0 || index >= this.totalSlides) return;
    
    this.slides.forEach((slide, i) => {
      slide.classList.toggle('active', i === index);
    });

    this.currentIndex = index;

    // Update Counter & Progress Bar
    const pad = (n) => String(n).padStart(2, '0');
    this.slideCounter.textContent = `${pad(index + 1)} / ${pad(this.totalSlides)}`;
    const progressPct = ((index + 1) / this.totalSlides) * 100;
    this.progressBar.style.width = `${progressPct}%`;

    // Button States
    this.btnPrev.disabled = index === 0;
    this.btnNext.disabled = index === this.totalSlides - 1;

    // Update Speaker Notes
    this.updateSpeakerNotes(index);

    // Update Overview Cards Active State
    document.querySelectorAll('.overview-card').forEach((card, i) => {
      card.classList.toggle('active', i === index);
    });
  }

  nextSlide() {
    if (this.currentIndex < this.totalSlides - 1) {
      this.renderSlide(this.currentIndex + 1);
    }
  }

  prevSlide() {
    if (this.currentIndex > 0) {
      this.renderSlide(this.currentIndex - 1);
    }
  }

  goToSlide(index) {
    this.renderSlide(index);
    if (this.overviewModal.classList.contains('open')) {
      this.toggleOverview(false);
    }
  }

  updateSpeakerNotes(index) {
    const currentSlide = this.slides[index];
    const notesEl = currentSlide.querySelector('.speaker-notes');
    if (notesEl) {
      this.presenterContent.innerHTML = notesEl.innerHTML;
    } else {
      this.presenterContent.innerHTML = `<p><em>No speaker notes registered for Slide ${index + 1}.</em></p>`;
    }
  }

  togglePresenterDrawer(forceState) {
    this.isPresenterOpen = (typeof forceState === 'boolean') ? forceState : !this.isPresenterOpen;
    this.presenterDrawer.classList.toggle('open', this.isPresenterOpen);
    document.getElementById('btnNotes').classList.toggle('active', this.isPresenterOpen);
  }

  toggleOverview(forceState) {
    const isOpen = (typeof forceState === 'boolean') ? forceState : !this.overviewModal.classList.contains('open');
    this.overviewModal.classList.toggle('open', isOpen);
    document.getElementById('btnOverview').classList.toggle('active', isOpen);
  }

  buildOverviewGrid() {
    this.overviewGrid.innerHTML = '';
    this.slides.forEach((slide, index) => {
      const titleEl = slide.querySelector('.slide-title');
      const title = titleEl ? titleEl.innerText : `Slide ${index + 1}`;
      
      const card = document.createElement('div');
      card.className = `overview-card ${index === this.currentIndex ? 'active' : ''}`;
      card.innerHTML = `
        <div class="overview-num">SLIDE ${String(index + 1).padStart(2, '0')}</div>
        <div class="overview-slide-title">${title}</div>
      `;
      card.addEventListener('click', () => this.goToSlide(index));
      this.overviewGrid.appendChild(card);
    });
  }

  startTimer() {
    this.timerInterval = setInterval(() => {
      this.timerSeconds++;
      const mins = Math.floor(this.timerSeconds / 60);
      const secs = this.timerSeconds % 60;
      this.timerDisplay.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }, 1000);
  }

  toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => console.log(err));
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
  }
}

// Initialize on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  window.deck = new SlideDeck();
});
