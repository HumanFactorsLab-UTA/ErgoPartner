/**
 * Inline icons, drawn on a 24px grid at 1.6px stroke.
 *
 * Inline rather than an icon package: the set is small, every glyph here means
 * one specific thing in this product, and shipping a whole library to use nine
 * of its icons is weight for nothing.
 */
import type { SVGProps } from "react";

type Props = SVGProps<SVGSVGElement> & { size?: number };

function Base({ size = 16, children, ...rest }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconRecord = (p: Props) => (
  <Base {...p}>
    <circle cx="12" cy="12" r="8" />
    <circle cx="12" cy="12" r="3.2" fill="currentColor" stroke="none" />
  </Base>
);

export const IconLibrary = (p: Props) => (
  <Base {...p}>
    <path d="M4 5v14M9 5v14M14 6l5 13" />
    <path d="M3 19h18" />
  </Base>
);

export const IconFolder = (p: Props) => (
  <Base {...p}>
    <path d="M3 7V5a1 1 0 0 1 1-1h5l2 3h9a1 1 0 0 1 1 1v11H3Z" />
  </Base>
);

export const IconPerson = (p: Props) => (
  <Base {...p}>
    <circle cx="12" cy="8" r="3.4" />
    <path d="M5 20a7 7 0 0 1 14 0" />
  </Base>
);

export const IconTimeline = (p: Props) => (
  <Base {...p}>
    <path d="M3 17l5-6 4 3 4-7 5 5" />
    <path d="M3 21h18" />
  </Base>
);

export const IconAlert = (p: Props) => (
  <Base {...p}>
    <path d="M12 4.5 2.8 20h18.4L12 4.5Z" />
    <path d="M12 10v4.2M12 17.2h.01" />
  </Base>
);

export const IconVideo = (p: Props) => (
  <Base {...p}>
    <rect x="2.5" y="6" width="13" height="12" rx="2.5" />
    <path d="m15.5 11 6-3.2v8.4l-6-3.2Z" />
  </Base>
);

export const IconFrames = (p: Props) => (
  <Base {...p}>
    <rect x="3" y="5" width="8" height="14" rx="1.6" />
    <rect x="13" y="5" width="8" height="6" rx="1.6" />
    <rect x="13" y="13" width="8" height="6" rx="1.6" />
  </Base>
);

export const IconReport = (p: Props) => (
  <Base {...p}>
    <path d="M6 3h8l4 4v14H6z" />
    <path d="M14 3v4h4M9.5 13h5M9.5 16.5h5" />
  </Base>
);

export const IconSearch = (p: Props) => (
  <Base {...p}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4.5 4.5" />
  </Base>
);

export const IconChevron = (p: Props) => (
  <Base {...p}>
    <path d="m9 5 7 7-7 7" />
  </Base>
);

export const IconArrowRight = (p: Props) => (
  <Base {...p}>
    <path d="M4 12h15M13 6l6 6-6 6" />
  </Base>
);

export const IconArrowLeft = (p: Props) => (
  <Base {...p}>
    <path d="M20 12H5M11 6l-6 6 6 6" />
  </Base>
);

export const IconPlay = (p: Props) => (
  <Base {...p}>
    <path d="M7 4.8 19 12 7 19.2Z" fill="currentColor" />
  </Base>
);

export const IconPause = (p: Props) => (
  <Base {...p}>
    <rect x="7" y="5" width="3.6" height="14" rx="1.1" fill="currentColor" stroke="none" />
    <rect x="13.4" y="5" width="3.6" height="14" rx="1.1" fill="currentColor" stroke="none" />
  </Base>
);

export const IconTarget = (p: Props) => (
  <Base {...p}>
    <circle cx="12" cy="12" r="8" />
    <circle cx="12" cy="12" r="3" />
    <path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3" />
  </Base>
);

export const IconBody = (p: Props) => (
  <Base {...p}>
    <circle cx="12" cy="4.6" r="2.2" />
    <path d="M12 7v7M6.5 9.5h11M12 14l-3 7M12 14l3 7" />
  </Base>
);

export const IconGauge = (p: Props) => (
  <Base {...p}>
    <path d="M4 17a8 8 0 1 1 16 0" />
    <path d="m12 17 4-5" />
  </Base>
);

export const IconInfo = (p: Props) => (
  <Base {...p}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 11v5M12 8h.01" />
  </Base>
);

export const IconPdf = (p: Props) => (
  <Base {...p}>
    <path d="M6 3h8l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
    <path d="M14 3v5h5" />
    <path d="M8 13h4" />
    <path d="M8 17h6" />
  </Base>
);
