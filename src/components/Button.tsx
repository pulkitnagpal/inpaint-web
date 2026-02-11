import { ReactNode, useState } from 'react'

interface ButtonProps {
  children: ReactNode
  className?: string
  icon?: ReactNode
  primary?: boolean
  disabled?: boolean
  style?: {
    [key: string]: string
  }
  onClick?: () => void
  onDown?: () => void
  onUp?: () => void
  onEnter?: () => void
  onLeave?: () => void
}

export default function Button(props: ButtonProps) {
  const {
    children,
    className,
    icon,
    primary,
    disabled,
    style,
    onClick,
    onDown,
    onUp,
    onEnter,
    onLeave,
  } = props
  const [active, setActive] = useState(false)
  let background = ''
  if (disabled) {
    background = 'bg-gray-300 text-gray-500 cursor-not-allowed'
  } else if (primary) {
    background = 'bg-primary hover:bg-black hover:text-white'
  } else if (active) {
    background = 'bg-black text-white'
  } else {
    background = 'hover:bg-primary'
  }
  return (
    <div
      role="button"
      onKeyDown={() => {
        if (!disabled) onDown?.()
      }}
      onClick={() => {
        if (!disabled) onClick?.()
      }}
      onPointerDown={() => {
        if (!disabled) {
          setActive(true)
          onDown?.()
        }
      }}
      onPointerUp={() => {
        if (!disabled) {
          setActive(false)
          onUp?.()
        }
      }}
      onPointerEnter={() => {
        if (!disabled) onEnter?.()
      }}
      onPointerLeave={() => {
        if (!disabled) onLeave?.()
      }}
      tabIndex={disabled ? -1 : -1}
      className={[
        'inline-flex space-x-3 py-3 px-5 rounded-md',
        disabled ? 'cursor-not-allowed' : 'cursor-pointer',
        background,
        className,
      ].join(' ')}
      style={style}
    >
      {icon}
      <span className="whitespace-nowrap select-none">{children}</span>
    </div>
  )
}
